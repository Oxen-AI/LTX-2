import logging
from collections.abc import Iterator

import torch

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.guiders import (
    MultiModalGuiderFactory,
    MultiModalGuiderParams,
    create_multimodal_guider_factory,
)
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.protocols import DiffusionStepProtocol
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
from ltx_core.model.upsampler import upsample_video
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.model.video_vae import decode_video as vae_decode_video
from ltx_core.quantization import QuantizationPolicy
from ltx_core.types import Audio, LatentState, VideoPixelShape
from ltx_pipelines.utils import (
    ModelLedger,
    assert_resolution,
    cleanup_memory,
    combined_image_conditionings,
    denoise_audio_video,
    encode_prompts,
    euler_denoising_loop,
    get_device,
    multi_modal_guider_factory_denoising_func,
    simple_denoising_func,
)
from ltx_pipelines.utils.args import ImageConditioningInput, default_2_stage_arg_parser, detect_checkpoint_path
from ltx_pipelines.utils.constants import STAGE_2_DISTILLED_SIGMA_VALUES, detect_params
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.types import PipelineComponents

from time import time

device = get_device()


class TI2VidTwoStagesPipeline:
    """
    Two-stage text/image-to-video generation pipeline.
    Stage 1 generates video at half of the target resolution with CFG guidance (assuming
    full model is used), then Stage 2 upsamples by 2x and refines using a distilled
    LoRA for higher quality output. Supports optional image conditioning via the
    images parameter.
    """

    def __init__(
        self,
        checkpoint_path: str,
        distilled_lora: list[LoraPathStrengthAndSDOps],
        spatial_upsampler_path: str,
        gemma_root: str,
        loras: list[LoraPathStrengthAndSDOps],
        device: torch.device = device,
        quantization: QuantizationPolicy | None = None,
    ):
        self.device = device
        self.dtype = torch.bfloat16
        self.stage_1_model_ledger = ModelLedger(
            dtype=self.dtype,
            device=device,
            checkpoint_path=checkpoint_path,
            gemma_root_path=gemma_root,
            spatial_upsampler_path=spatial_upsampler_path,
            loras=loras,
            quantization=quantization,
        )

        self.stage_2_model_ledger = self.stage_1_model_ledger.with_additional_loras(
            loras=distilled_lora,
        )

        self.pipeline_components = PipelineComponents(
            dtype=self.dtype,
            device=device,
        )

    def __call__(  # noqa: PLR0913
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        num_inference_steps: int,
        video_guider_params: MultiModalGuiderParams | MultiModalGuiderFactory,
        audio_guider_params: MultiModalGuiderParams | MultiModalGuiderFactory,
        images: list[ImageConditioningInput],
        tiling_config: TilingConfig | None = None,
        enhance_prompt: bool = False,
    ) -> tuple[Iterator[torch.Tensor], Audio]:
        assert_resolution(height=height, width=width, is_two_stage=True)

        print("Init...")
        start_time = time()
        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()
        dtype = torch.bfloat16
        print(f"Init time: {time() - start_time:.3f}")

        print("Encode prompts...")
        start_time = time()
        ctx_p, ctx_n = encode_prompts(
            [prompt, negative_prompt],
            self.stage_1_model_ledger,
            enhance_first_prompt=enhance_prompt,
            enhance_prompt_image=images[0][0] if len(images) > 0 else None,
            enhance_prompt_seed=seed,
        )
        v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
        v_context_n, a_context_n = ctx_n.video_encoding, ctx_n.audio_encoding
        print(f"Encode time: {time() - start_time:.3f}")

        print("Init video encoder...")
        start_time = time()
        # Stage 1: encode image conditionings with the VAE encoder, then free it
        # before loading the transformer to reduce peak VRAM.
        stage_1_output_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=width // 2,
            height=height // 2,
            fps=frame_rate,
        )
        video_encoder = self.stage_1_model_ledger.video_encoder()
        print(f"Init video encoder time: {time() - start_time:.3f}")
        print("Stage 1 conditionings...")
        start_time = time()
        stage_1_conditionings = combined_image_conditionings(
            images=images,
            height=stage_1_output_shape.height,
            width=stage_1_output_shape.width,
            video_encoder=video_encoder,
            dtype=dtype,
            device=self.device,
        )
        torch.cuda.synchronize()
        print(f"Stage 1 conditionings time: {time() - start_time:.3f}")
        print("Clean up memory...")
        start_time = time()
        del video_encoder
        cleanup_memory()
        print(f"Clean up memory time: {time() - start_time:.3f}")

        print("Init transformer...")
        start_time = time()
        transformer = self.stage_1_model_ledger.transformer()
        sigmas = LTX2Scheduler().execute(steps=num_inference_steps).to(dtype=torch.float32, device=self.device)
        print(f"Init transformer time: {time() - start_time:.3f}")

        def first_stage_denoising_loop(
            sigmas: torch.Tensor, video_state: LatentState, audio_state: LatentState, stepper: DiffusionStepProtocol
        ) -> tuple[LatentState, LatentState]:
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=multi_modal_guider_factory_denoising_func(
                    video_guider_factory=create_multimodal_guider_factory(
                        params=video_guider_params,
                        negative_context=v_context_n,
                    ),
                    audio_guider_factory=create_multimodal_guider_factory(
                        params=audio_guider_params,
                        negative_context=a_context_n,
                    ),
                    v_context=v_context_p,
                    a_context=a_context_p,
                    transformer=transformer,  # noqa: F821
                ),
            )

        print("Denoise...")
        start_time = time()
        video_state, audio_state = denoise_audio_video(
            output_shape=stage_1_output_shape,
            conditionings=stage_1_conditionings,
            noiser=noiser,
            sigmas=sigmas,
            stepper=stepper,
            denoising_loop_fn=first_stage_denoising_loop,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
        )

        torch.cuda.synchronize()
        print(f"Denoise time: {time() - start_time:.3f}")
        print("Clean up memory...")
        start_time = time()
        del transformer
        cleanup_memory()
        print(f"Clear up memory time: {time() - start_time:.3f}")

        print("Stage 2 load video encoder...")
        start_time = time()
        # Stage 2: Upsample and refine the video at higher resolution with distilled LORA.
        video_encoder = self.stage_1_model_ledger.video_encoder()
        print(f"Stage 2 load video encoder time: {time() - start_time:.3f}")
        upscaled_video_latent = upsample_video(
            latent=video_state.latent[:1],
            video_encoder=video_encoder,
            upsampler=self.stage_2_model_ledger.spatial_upsampler(),
        )

        print("Stage 2 image conditionings...")
        start_time = time()
        stage_2_output_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=frame_rate)
        stage_2_conditionings = combined_image_conditionings(
            images=images,
            height=stage_2_output_shape.height,
            width=stage_2_output_shape.width,
            video_encoder=video_encoder,
            dtype=dtype,
            device=self.device,
        )
        del video_encoder
        torch.cuda.synchronize()
        cleanup_memory()
        print(f"Stage 2 image conditionings time: {time() - start_time:.3f}")

        print("Init stage 2 transformer...")
        start_time = time()
        transformer = self.stage_2_model_ledger.transformer()
        distilled_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(self.device)
        print(f"Init stage 2 transformer time: {time() - start_time:.3f}")

        def second_stage_denoising_loop(
            sigmas: torch.Tensor, video_state: LatentState, audio_state: LatentState, stepper: DiffusionStepProtocol
        ) -> tuple[LatentState, LatentState]:
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=simple_denoising_func(
                    video_context=v_context_p,
                    audio_context=a_context_p,
                    transformer=transformer,  # noqa: F821
                ),
            )

        print("Stage 2 Denoise")
        start_time = time()
        video_state, audio_state = denoise_audio_video(
            output_shape=stage_2_output_shape,
            conditionings=stage_2_conditionings,
            noiser=noiser,
            sigmas=distilled_sigmas,
            stepper=stepper,
            denoising_loop_fn=second_stage_denoising_loop,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            noise_scale=distilled_sigmas[0],
            initial_video_latent=upscaled_video_latent,
            initial_audio_latent=audio_state.latent,
        )

        torch.cuda.synchronize()
        print(f"Stage 2 Denoise time: {time() - start_time:.3f}")
        print("Stage 2 cleanup...")
        start_time = time()
        del transformer
        cleanup_memory()
        print(f"Stage 2 cleanup time: {time() - start_time:.3f}")

        print("Decode video...")
        start_time = time()
        decoded_video = vae_decode_video(
            video_state.latent, self.stage_2_model_ledger.video_decoder(), tiling_config, generator
        )
        print(f"Decode video time: {time() - start_time:.3f}")
        print("Decode audio...")
        start_time = time()
        decoded_audio = vae_decode_audio(
            audio_state.latent, self.stage_2_model_ledger.audio_decoder(), self.stage_2_model_ledger.vocoder()
        )
        print(f"Decode audio time: {time() - start_time:.3f}")
        return decoded_video, decoded_audio


@torch.inference_mode()
def main() -> None:
    logging.getLogger().setLevel(logging.INFO)
    checkpoint_path = detect_checkpoint_path()
    params = detect_params(checkpoint_path)
    parser = default_2_stage_arg_parser(params=params)
    args = parser.parse_args()
    pipeline = TI2VidTwoStagesPipeline(
        checkpoint_path=args.checkpoint_path,
        distilled_lora=args.distilled_lora,
        spatial_upsampler_path=args.spatial_upsampler_path,
        gemma_root=args.gemma_root,
        loras=tuple(args.lora) if args.lora else (),
        quantization=args.quantization,
    )
    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(args.num_frames, tiling_config)
    video, audio = pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        num_inference_steps=args.num_inference_steps,
        video_guider_params=MultiModalGuiderParams(
            cfg_scale=args.video_cfg_guidance_scale,
            stg_scale=args.video_stg_guidance_scale,
            rescale_scale=args.video_rescale_scale,
            modality_scale=args.a2v_guidance_scale,
            skip_step=args.video_skip_step,
            stg_blocks=args.video_stg_blocks,
        ),
        audio_guider_params=MultiModalGuiderParams(
            cfg_scale=args.audio_cfg_guidance_scale,
            stg_scale=args.audio_stg_guidance_scale,
            rescale_scale=args.audio_rescale_scale,
            modality_scale=args.v2a_guidance_scale,
            skip_step=args.audio_skip_step,
            stg_blocks=args.audio_stg_blocks,
        ),
        images=args.images,
        tiling_config=tiling_config,
    )

    encode_video(
        video=video,
        fps=args.frame_rate,
        audio=audio,
        output_path=args.output_path,
        video_chunks_number=video_chunks_number,
    )


if __name__ == "__main__":
    main()
