import torch, os, gc
import tensorrt as trt
from PIL import Image
from utilities import TRT_LOGGER
from txt2img_xl_pipeline import Txt2ImgXLPipeline
from img2img_xl_pipeline import Img2ImgXLPipeline

default_args = {
    'scheduler': "DDIM",
    'negative_prompt': "",
    'denoising_steps': 30,
    'output_dir': "output",
    'height': 1024,
    'width': 1024,
    'guidance': 7.0,
    'seed': None,
    'hf_token': None,
    'verbose': False,
    'framework_model_dir': "pytorch_model",
    'onnx_opset': 17,
    'onnx_base_dir': "onnx-ckpt/sdxl-1.0-base",
    'onnx_refiner_dir': "onnx-ckpt/sdxl-1.0-refiner",
    'engine_base_dir': "trt-engine/xl_base",
    'engine_refiner_dir': "trt-engine/xl_refiner",
    'num_warmup_runs': 5,
    'timing_cache': None,
    'build_static_batch': True,
    'build_dynamic_shape': False,
    'build_preview_features': False,
    'build_all_tactics': False,
}

class StableDiffusionXLInferer:
    def __init__(self, **kwargs):
        self.args = default_args.copy()
        self.args.update(kwargs)
        trt.init_libnvinfer_plugins(TRT_LOGGER, "")
        self.base_pipeline = None
        self.refiner_pipeline = None

    def load_base_pipeline(self):
        if self.base_pipeline is None:
            self.base_pipeline = self.init_pipeline(Txt2ImgXLPipeline, False, self.args['onnx_base_dir'], self.args['engine_base_dir'])
        return self.base_pipeline

    def load_refiner_pipeline(self):
        if self.refiner_pipeline is None:
            self.refiner_pipeline = self.init_pipeline(Img2ImgXLPipeline, True, self.args['onnx_refiner_dir'], self.args['engine_refiner_dir'])
        return self.refiner_pipeline

    def init_pipeline(self, pipeline_class, refiner, onnx_dir, engine_dir):
        pipeline = pipeline_class(
            scheduler=self.args['scheduler'],
            denoising_steps=self.args['denoising_steps'],
            output_dir=self.args['output_dir'],
            version="xl-1.0",
            hf_token=self.args['hf_token'],
            verbose=self.args['verbose'],
            max_batch_size=1,
            use_cuda_graph=False,
            refiner=refiner,
            framework_model_dir=self.args['framework_model_dir']
        )
        pipeline.loadEngines(
            engine_dir=engine_dir,
            framework_model_dir=self.args['framework_model_dir'],
            onnx_dir=onnx_dir,
            onnx_opset=self.args['onnx_opset'],
            opt_batch_size=1,
            opt_image_height=self.args['height'],
            opt_image_width=self.args['width'],
            static_batch=self.args['build_static_batch'],
            static_shape=not self.args['build_dynamic_shape'],
            enable_preview=self.args['build_preview_features'],
            enable_all_tactics=self.args['build_all_tactics'],
            timing_cache=self.args['timing_cache']
        )
        pipeline.activateEngines()
        pipeline.loadResources(self.args['height'], self.args['width'], 1, self.args['seed'])
        return pipeline

    def tensor_to_pil(self, tensor):
        # Assumes that tensor is in CxHxW format with values in range [-1, 1]
        tensor = tensor.add(1).div(2).clamp(0, 1)  # Denormalize from [-1, 1] to [0, 1]
        tensor = tensor.mul(255).byte()  # Scale to [0, 255] and convert to uint8
        tensor = tensor.cpu().permute(1, 2, 0).numpy()  # Rearrange from CxHxW to HxWxC and convert to NumPy array
        return Image.fromarray(tensor)  # 
    
    def batch_infer(self, prompts, use_refiner=False):
        # Charge le modèle de base et génère les images latentes
        base_pipeline = self.load_base_pipeline()
        images = []
        for prompt in prompts:
            img, _ = base_pipeline.infer(
                [prompt],  # Utilise le bon prompt pour chaque image
                [self.args['negative_prompt']],
                self.args['height'],
                self.args['width'],
                self.args['guidance'],
                self.args['seed'],
                warmup=False,
                verbose=self.args['verbose'],
                return_type="latents" if use_refiner else "image_nosave"
            )
            images.append(img)

        self.base_pipeline.teardown()
        del base_pipeline
        del self.base_pipeline
        gc.collect()
        torch.cuda.empty_cache()

        if use_refiner:
            refiner_pipeline = self.load_refiner_pipeline()
            refined_images = []
            for prompt, img in zip(prompts, images):
                refined_img, _ = refiner_pipeline.infer(
                    [prompt],
                    [self.args['negative_prompt']],
                    img,
                    self.args['height'],
                    self.args['width'],
                    self.args['guidance'],
                    self.args['seed'],
                    warmup=False,
                    verbose=self.args['verbose'],
                    return_type="image_nosave"
                )
                refined_images.append(refined_img)
            images = refined_images
            refiner_pipeline.teardown()
            del refiner_pipeline
            gc.collect()
            torch.cuda.empty_cache()

        pil_images = [self.tensor_to_pil(image.squeeze(0)) for image in images]
        return pil_images

if __name__ == '__main__':
    new_prompts = ['Olivia, a 23-year-old, bumps into Wilson Alexander, CEO of Alexander Corporation, while job hunting. He offers her a job and a lunch interview.', 'Olivia, a 23-year-old, bumps into Wilson Alexander, CEO of Alexander Corporation, while job hunting. He offers her a job and a lunch interview. She is excited but wary. They have a history, with Olivia taking care of Wilson for a year. He is diagnosed with a rare heart disease and Olivia is scared for him. They have a heart-to-heart conversation where Olivia expresses her feelings for Wilson, who is ashamed of his fragility.', 'Olivia, a 23-year-old, bumps into Wilson Alexander, CEO of Alexander Corporation, while job hunting. He offers her a job and a lunch interview. They have a history, with Olivia taking care of Wilson for a year. They have a heart-to-heart conversation where Olivia expresses her feelings for Wilson, who is ashamed of his fragility. Wilson dies two weeks later from heart failure while sleeping. Olivia has one last meal with him before he dies.']
    inferer = StableDiffusionXLInferer()
    # images = inferer.batch_infer(prompts=['Olivia, a 23-year-old, bumps into Wilson Alexander, CEO of Alexander Corporation, while job hunting. He offers her a job and a lunch interview.', 'A futuristic cityscape'], use_refiner=True)
    images = inferer.batch_infer(prompts=new_prompts, use_refiner=True)
    for img in images:
        img.show()