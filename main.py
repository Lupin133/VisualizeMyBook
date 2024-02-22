import argparse, time, gc, torch, multiprocessing
from pdf_logic import get_n_text_sections, insert_images_and_save
from LLMinfer import LLMInference
from sdxl_infer import StableDiffusionXLInferer

def summarize_pages(conn, pdf_path, frequency):
    text_sections = get_n_text_sections(pdf_path, frequency)

    inference_engine = LLMInference(
        max_output_len=150,
        tokenizer_dir="Mistral-7B-Instruct-v0.1",
        engine_dir="converted",
        max_attention_window_size=4096
    )

    print("Running Summary inference...")
    prompts = []
    for text_section in text_sections:
        full_prompt_template = f"""
            TEXT:
            {text_section}
            ENDTEXT

            Task: Taking into account the characters and themes present in earlier prompts, condense the current lengthy text into a succinct summary within 50 words. This summary is for guiding a Stable Diffusion model to create a visual image. It should reflect the essence and continuity of the main narrative without directly duplicating previous prompts. Craft a summary that captures the essence of the text while creatively integrating known characters in a way that is ideal for visual depiction.

            Summary:
        """
        response = inference_engine.run_inference(full_prompt_template)
        only_response = response.split("Summary:")[1].strip()
        prompts.append(only_response)
    
    print("Number of prompts:", len(prompts))
    print("Prompts:", prompts)

    del inference_engine
    gc.collect()
    torch.cuda.empty_cache()

    conn.send(prompts)
    conn.close()

# It seems that using SDXLTensorRT+TensorRT-LLM don't work together (i get random noise images), using multiple processes to run them completly separately seems to fix the issue
def run_summarize_pages(pdf_path, frequency):
    parent_conn, child_conn = multiprocessing.Pipe()
    test_process = multiprocessing.Process(target=summarize_pages, args=(child_conn, pdf_path, frequency))

    test_process.start()
    test_process.join()

    find_prompts = parent_conn.recv()

    return find_prompts


def main():
    parser = argparse.ArgumentParser(description='Process a PDF by adding images and extracting text.')
    parser.add_argument('--pdf_path', type=str, help='Path to the PDF file to be processed.')
    parser.add_argument('--frequency', default=5, type=int, help='Frequency of inserting images (every n pages).')
    parser.add_argument('--use_refiner', action='store_true', help='Use the image refiner.')

    args = parser.parse_args()

    text_sections = get_n_text_sections(args.pdf_path, args.frequency)
    find_prompts = run_summarize_pages(args.pdf_path, args.frequency)
    print("Prompts created : ", find_prompts)

    inferer = StableDiffusionXLInferer()
    images = inferer.batch_infer(prompts=find_prompts, use_refiner=True)

    insert_images_and_save(args.pdf_path, images, args.frequency, "final.pdf")

if __name__ == '__main__':
    main()
