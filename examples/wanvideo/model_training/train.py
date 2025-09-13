import torch, os, json, copy
from diffsynth import load_state_dict
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, launch_training_task, wan_parser, DPOVideoDataset
from diffsynth.trainers.unified_dataset import UnifiedDataset
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import wandb

wandb.login(key="881a9b14affd90a6fe2d60376ba0f08be5a6bee8")


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32, lora_checkpoint=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        dpo_beta=0.1,
    ):
        super().__init__()
        self.last_metrics = {}
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, enable_fp8_training=False)
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs)

        # Prepare a frozen reference model (LoRA-free) before injecting LoRA
        self.reference_model = None
        self.lora_base_model_name = None
        if lora_base_model is not None and hasattr(self.pipe, lora_base_model):
            self.lora_base_model_name = lora_base_model
            base_model = getattr(self.pipe, lora_base_model)
            self.reference_model = copy.deepcopy(base_model)
            for p in self.reference_model.parameters():
                p.requires_grad = False

        # Training mode (inject LoRA into online model only)
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint=lora_checkpoint,
            enable_fp8_training=False,
        )
        
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        self.dpo_beta = dpo_beta
        
        
    def forward_preprocess(self, data):
        # CFG-sensitive parameters
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        
        # CFG-unsensitive parameters
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["win_video"],
            "height": data["win_video"][0].size[1],
            "width": data["win_video"][0].size[0],
            "num_frames": len(data["win_video"]),
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        
        # Extra inputs
        for extra_input in self.extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["win_video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["win_video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]
        
        # Pipeline units will automatically process the input parameters.
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}
    
    
    def forward(self, data, inputs=None):
        # Build two preprocessed inputs
        inputs_win = self.forward_preprocess({
            "prompt": data["prompt"],
            "win_video": data["win_video"],
            "lose_video": data["lose_video"],
        })
        inputs_lose = self.forward_preprocess({
            "prompt": data["prompt"],
            "win_video": data["lose_video"], # reuse keys
            "lose_video": data["lose_video"],
        })

        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}

        # Sample a single timestep for both samples
        max_timestep_boundary = int(inputs_win.get("max_timestep_boundary", 1) * self.pipe.scheduler.num_train_timesteps)
        min_timestep_boundary = int(inputs_win.get("min_timestep_boundary", 0) * self.pipe.scheduler.num_train_timesteps)
        timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(dtype=self.pipe.torch_dtype, device=self.pipe.device)

        # Prepare latents/noise and training target
        # Note: requires_grad should be True for the online model's input
        latents_win = self.pipe.scheduler.add_noise(
            inputs_win["input_latents"], inputs_win["noise"], timestep
        )
        latents_lose = self.pipe.scheduler.add_noise(
            inputs_lose["input_latents"], inputs_win["noise"], timestep
        )
        training_target_win = self.pipe.scheduler.training_target(
            inputs_win["input_latents"], inputs_win["noise"], timestep
        )
        training_target_lose = self.pipe.scheduler.training_target(
            inputs_lose["input_latents"], inputs_win["noise"], timestep
        )
        weight = self.pipe.scheduler.training_weight(timestep)

        # Online (LoRA-enabled) forward
        model_inputs = {}
        for k in inputs_win:
            if isinstance(inputs_win[k], torch.Tensor):
                try:
                    model_inputs[k] = torch.cat([inputs_win[k], inputs_lose[k]], dim=0)
                except:
                    model_inputs[k] = inputs_win[k]
            else:
                model_inputs[k] = inputs_win[k]
        model_inputs["latents"] = torch.cat([latents_win, latents_lose], dim=0).requires_grad_()

        noise_pred_theta = self.pipe.model_fn(**models, **model_inputs, timestep=timestep)
        noise_pred_theta_win, noise_pred_theta_lose = torch.chunk(noise_pred_theta, 2, dim=0)
        # mse_theta_win = torch.nn.functional.mse_loss(noise_pred_theta_win.float(), training_target_win.float(), reduction="none").mean(dim=list(range(1, training_target_win.ndim)))
        # mse_theta_lose = torch.nn.functional.mse_loss(noise_pred_theta_lose.float(), training_target_lose.float(), reduction="none").mean(dim=list(range(1, training_target_lose.ndim)))
        # win_loss_theta, lose_loss_theta = mse_theta_win * weight, mse_theta_lose * weight
        mse_theta_win = ((noise_pred_theta_win.float() - training_target_win.float()) ** 2).reshape(noise_pred_theta_win.shape[0], -1).mean(dim=1)
        mse_theta_lose = ((noise_pred_theta_lose.float() - training_target_lose.float()) ** 2).reshape(noise_pred_theta_lose.shape[0], -1).mean(dim=1)
        # del noise_pred_theta_win, noise_pred_theta_lose
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        # Frozen/reference (separate cloned model without LoRA)
        with torch.no_grad():
            model_inputs["latents"] = torch.cat([latents_win, latents_lose], dim=0)
            models_ref = dict(models)
            if self.reference_model is not None and self.lora_base_model_name in models_ref:
                models_ref[self.lora_base_model_name] = self.reference_model
            noise_pred_old = self.pipe.model_fn(**models_ref, **model_inputs, timestep=timestep)
            noise_pred_old_win, noise_pred_old_lose = torch.chunk(noise_pred_old, 2, dim=0)
        # mse_old_win = torch.nn.functional.mse_loss(noise_pred_old_win.float(), training_target_win.float(), reduction="none").mean(dim=list(range(1, training_target_win.ndim)))
        # mse_old_lose = torch.nn.functional.mse_loss(noise_pred_old_lose.float(), training_target_lose.float(), reduction="none").mean(dim=list(range(1, training_target_lose.ndim)))
        # win_loss_old, lose_loss_old = mse_old_win * weight, mse_old_lose * weight
        mse_old_win = ((noise_pred_old_win.float() - training_target_win.float()) ** 2).reshape(noise_pred_old_win.shape[0], -1).mean(dim=1)
        mse_old_lose = ((noise_pred_old_lose.float() - training_target_lose.float()) ** 2).reshape(noise_pred_old_lose.shape[0], -1).mean(dim=1)
        # del noise_pred_old_win, noise_pred_old_lose
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        # DPO loss
        w_diff = mse_theta_win - mse_old_win.detach()
        l_diff = mse_theta_lose - mse_old_lose.detach()
        w_l_diff = w_diff - l_diff
        inside_term = -0.5 * self.dpo_beta * w_l_diff
        # inside_term = -0.5 * w_l_diff
        loss = -torch.nn.functional.logsigmoid(inside_term).mean()
        # Aggregate scalar metrics (handle possible vector tensors)
        m_theta_win = mse_theta_win.mean().detach().float().item()
        m_theta_lose = mse_theta_lose.mean().detach().float().item()
        m_old_win = mse_old_win.mean().detach().float().item()
        m_old_lose = mse_old_lose.mean().detach().float().item()
        dpo_loss = loss.detach().float().item()
        self.last_metrics = {
            "loss/dpo": dpo_loss,
            "loss/mse_theta_win": m_theta_win,
            "loss/mse_theta_lose": m_theta_lose,
            "loss/mse_old_win": m_old_win,
            "loss/mse_old_lose": m_old_lose,
        }
        # flush cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return loss


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    # dataset = UnifiedDataset(
    #     base_path=args.dataset_base_path,
    #     metadata_path=args.dataset_metadata_path,
    #     repeat=args.dataset_repeat,
    #     data_file_keys=args.data_file_keys.split(","),
    #     main_data_operator=UnifiedDataset.default_video_operator(
    #         base_path=args.dataset_base_path,
    #         max_pixels=args.max_pixels,
    #         height=args.height,
    #         width=args.width,
    #         height_division_factor=16,
    #         width_division_factor=16,
    #         num_frames=args.num_frames,
    #         time_division_factor=4,
    #         time_division_remainder=1,
    #     ),
    # )
    dataset = DPOVideoDataset(args=args)
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt
    )
    launch_training_task(dataset, model, model_logger, args=args)
