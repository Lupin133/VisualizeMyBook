import gradio as gr
from gradio import components
import multiprocessing
from pdf_logic import get_n_text_sections, insert_images_and_save
from LLMinfer import LLMInference
from sdxl_infer import StableDiffusionXLInferer
from main import run_summarize_pages

def process_pdf(pdf_path, frequency, use_refiner):
    text_sections = get_n_text_sections(pdf_path, frequency)
    gr.Info("Running Summary inference...")
    find_prompts = run_summarize_pages(pdf_path, frequency)
    gr.Info(f"Created {len(find_prompts)} summaries.")
    inferer = StableDiffusionXLInferer()
    images = inferer.batch_infer(prompts=find_prompts, use_refiner=use_refiner)
    gr.Info("Inserting images and saving...")
    insert_images_and_save(pdf_path, images, frequency, "final.pdf")
    gr.Info("Process complete.")
    return "final.pdf"

gradioInterface = gr.Interface(
    fn=process_pdf,
    inputs=[
        gr.File(label="Upload PDF"),
        gr.Slider(minimum=3, maximum=25, step=1, value=5, label="Frequency of Images"),
        gr.Checkbox(label="Use Image Refiner", value=True),
    ],
    outputs=gr.File(label="Download Processed PDF"),
    title="Visualize My Book 📖",
    description=(
        "This tool processes a PDF file, summarizing pages with an AI model, "
        "and then uses those summaries to prompt a Stable Diffusion model to "
        "create images. These images are then inserted back into the PDF. "
        "Adjust the frequency to control how often images are inserted."
    ),
    allow_flagging=False
)

if __name__ == '__main__':
    gradioInterface.launch()