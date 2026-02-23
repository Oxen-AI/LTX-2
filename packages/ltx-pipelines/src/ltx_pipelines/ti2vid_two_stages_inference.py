"""Inference-optimized two-stage text/image-to-video generation pipeline.

Eagerly loads all models during initialization. Small models (text encoder,
VAE, upsampler, vocoder) remain on GPU permanently. Transformers are cached
in CPU memory and swapped to GPU per denoising stage.

Targets H100 80GB VRAM: ~10GB persistent models + ~38GB per-stage transformer.
"""

import logging
from collections.abc import Iterator

import torch

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.guiders import CFGGuider
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.protocols import DiffusionStepProtocol
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.loader.registry import StateDictRegistry
from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
from ltx_core.model.upsampler import upsample_video
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.model.video_vae import decode_video as vae_decode_video
from ltx_core.text_encoders.gemma import encode_text
from ltx_core.types import LatentState, VideoPixelShape
from ltx_pipelines.utils import ModelLedger
from ltx_pipelines.utils.args import default_2_stage_arg_parser
from ltx_pipelines.utils.constants import (
    AUDIO_SAMPLE_RATE,
    STAGE_2_DISTILLED_SIGMA_VALUES,
)
from ltx_pipelines.utils.helpers import (
    assert_resolution,
    cleanup_memory,
    denoise_audio_video,
    euler_denoising_loop,
    generate_enhanced_prompt,
    get_device,
    guider_denoising_func,
    image_conditionings_by_replacing_latent,
    simple_denoising_func,
)
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.types import PipelineComponents

device = get_device()


class TI2VidTwoStagesInferencePipeline:
    """
    Inference-optimized two-stage text/image-to-video generation pipeline.

    All models are eagerly loaded during initialization. Small models stay on
    GPU permanently; transformers are cached in CPU memory and swapped to GPU
    per stage. Same two-stage generation as TI2VidTwoStagesPipeline.
    """

    def __init__(
        self,
        checkpoint_path: str,
        distilled_lora: list[LoraPathStrengthAndSDOps],
        spatial_upsampler_path: str,
        gemma_root: str,
        loras: list[LoraPathStrengthAndSDOps],
        device: str = device,
        fp8transformer: bool = False,
    ):
        self.device = device
        self.dtype = torch.bfloat16

        # StateDictRegistry caches checkpoint state dicts in CPU memory so
        # each safetensors file is read from disk only once across all builders.
        registry = StateDictRegistry()

        self.stage_1_model_ledger = ModelLedger(
            dtype=self.dtype,
            device=device,
            checkpoint_path=checkpoint_path,
            gemma_root_path=gemma_root,
            spatial_upsampler_path=spatial_upsampler_path,
            loras=loras,
            registry=registry,
            fp8transformer=fp8transformer,
        )

        self.stage_2_model_ledger = self.stage_1_model_ledger.with_loras(
            loras=distilled_lora,
        )

        self.pipeline_components = PipelineComponents(
            dtype=self.dtype,
            device=device,
        )

        # --- Eagerly load all small models onto GPU ---
        print("Loading text encoder...")
        self._text_encoder = self.stage_1_model_ledger.text_encoder()

        print("Loading video encoder...")
        self._video_encoder = self.stage_1_model_ledger.video_encoder()

        print("Loading spatial upsampler...")
        self._spatial_upsampler = self.stage_2_model_ledger.spatial_upsampler()

        print("Loading video decoder...")
        self._video_decoder = self.stage_2_model_ledger.video_decoder()

        print("Loading audio decoder...")
        self._audio_decoder = self.stage_2_model_ledger.audio_decoder()

        print("Loading vocoder...")
        self._vocoder = self.stage_2_model_ledger.vocoder()

        # --- Build transformers with fused LoRAs, then cache on CPU ---
        # Stage 1: base LoRAs (user fine-tune)
        # Stage 2: base LoRAs + distilled LoRA (refinement)
        # Both are built once and kept in CPU memory. During predict, they are
        # swapped to GPU via .to(device) instead of rebuilding from disk.
        print("Building Stage 1 transformer (caching in CPU memory)...")
        self._stage_1_transformer = self.stage_1_model_ledger.transformer().cpu()
        cleanup_memory()

        print("Building Stage 2 transformer (caching in CPU memory)...")
        self._stage_2_transformer = self.stage_2_model_ledger.transformer().cpu()
        cleanup_memory()

        # Free the registry's checkpoint caches - we've built everything we need.
        registry.clear()

        print("Inference pipeline ready!")

    @torch.inference_mode()
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
        cfg_guidance_scale: float,
        images: list[tuple[str, int, float]],
        tiling_config: TilingConfig | None = None,
        enhance_prompt: bool = False,
    ) -> tuple[Iterator[torch.Tensor], torch.Tensor]:
        assert_resolution(height=height, width=width, is_two_stage=True)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()
        cfg_guider = CFGGuider(cfg_guidance_scale)
        dtype = torch.bfloat16

        # --- Text encoding (uses permanently-resident text encoder) ---
        if enhance_prompt:
            prompt = generate_enhanced_prompt(
                self._text_encoder,
                prompt,
                images[0][0] if len(images) > 0 else None,
                seed=seed,
            )
        context_p, context_n = encode_text(self._text_encoder, prompts=[prompt, negative_prompt])
        v_context_p, a_context_p = context_p
        v_context_n, a_context_n = context_n

        # --- Stage 1: Initial generation at half resolution ---
        # Swap Stage 1 transformer from CPU to GPU
        self._stage_1_transformer.to(self.device)
        transformer = self._stage_1_transformer

        sigmas = LTX2Scheduler().execute(steps=num_inference_steps).to(
            dtype=torch.float32, device=self.device
        )

        def first_stage_denoising_loop(
            sigmas: torch.Tensor,
            video_state: LatentState,
            audio_state: LatentState,
            stepper: DiffusionStepProtocol,
        ) -> tuple[LatentState, LatentState]:
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=guider_denoising_func(
                    cfg_guider,
                    v_context_p,
                    v_context_n,
                    a_context_p,
                    a_context_n,
                    transformer=transformer,
                ),
            )

        stage_1_output_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=width // 2,
            height=height // 2,
            fps=frame_rate,
        )
        stage_1_conditionings = image_conditionings_by_replacing_latent(
            images=images,
            height=stage_1_output_shape.height,
            width=stage_1_output_shape.width,
            video_encoder=self._video_encoder,
            dtype=dtype,
            device=self.device,
        )
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

        # Swap Stage 1 transformer back to CPU to free GPU memory for Stage 2
        torch.cuda.synchronize()
        self._stage_1_transformer.cpu()
        cleanup_memory()

        # --- Stage 2: Upsample and refine at full resolution ---
        upscaled_video_latent = upsample_video(
            latent=video_state.latent[:1],
            video_encoder=self._video_encoder,
            upsampler=self._spatial_upsampler,
        )

        torch.cuda.synchronize()
        cleanup_memory()

        # Swap Stage 2 transformer from CPU to GPU
        self._stage_2_transformer.to(self.device)
        transformer = self._stage_2_transformer
        distilled_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(self.device)

        def second_stage_denoising_loop(
            sigmas: torch.Tensor,
            video_state: LatentState,
            audio_state: LatentState,
            stepper: DiffusionStepProtocol,
        ) -> tuple[LatentState, LatentState]:
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=simple_denoising_func(
                    video_context=v_context_p,
                    audio_context=a_context_p,
                    transformer=transformer,
                ),
            )

        stage_2_output_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=frame_rate)
        stage_2_conditionings = image_conditionings_by_replacing_latent(
            images=images,
            height=stage_2_output_shape.height,
            width=stage_2_output_shape.width,
            video_encoder=self._video_encoder,
            dtype=dtype,
            device=self.device,
        )
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

        # Swap Stage 2 transformer back to CPU
        torch.cuda.synchronize()
        self._stage_2_transformer.cpu()
        cleanup_memory()

        # --- Decode using permanently-resident decoders ---
        decoded_video = vae_decode_video(
            video_state.latent, self._video_decoder, tiling_config, generator
        )
        decoded_audio = vae_decode_audio(
            audio_state.latent, self._audio_decoder, self._vocoder
        )

        return decoded_video, decoded_audio


@torch.inference_mode()
def main() -> None:
    logging.getLogger().setLevel(logging.INFO)
    parser = default_2_stage_arg_parser()
    args = parser.parse_args()
    pipeline = TI2VidTwoStagesInferencePipeline(
        checkpoint_path=args.checkpoint_path,
        distilled_lora=args.distilled_lora,
        spatial_upsampler_path=args.spatial_upsampler_path,
        gemma_root=args.gemma_root,
        loras=args.lora,
        fp8transformer=args.enable_fp8,
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
        cfg_guidance_scale=args.cfg_guidance_scale,
        images=args.images,
        tiling_config=tiling_config,
    )

    encode_video(
        video=video,
        fps=args.frame_rate,
        audio=audio,
        audio_sample_rate=AUDIO_SAMPLE_RATE,
        output_path=args.output_path,
        video_chunks_number=video_chunks_number,
    )


if __name__ == "__main__":
    main()
