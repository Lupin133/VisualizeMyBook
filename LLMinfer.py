# simplified version of the original run.py from TensorRT-LLM

import torch
from utils import load_tokenizer, read_model_name
from tensorrt_llm.runtime import ModelRunner  # Assumed to be the appropriate import based on the provided script

class LLMInference:
    def __init__(self, max_output_len, tokenizer_dir, engine_dir, max_attention_window_size):
        self.max_output_len = max_output_len
        self.tokenizer_dir = tokenizer_dir
        self.engine_dir = engine_dir
        self.max_attention_window_size = max_attention_window_size

        self.model_name = read_model_name(self.engine_dir)
        self.tokenizer, _, self.end_id = load_tokenizer(tokenizer_dir=self.tokenizer_dir, model_name=self.model_name)
        self.runner = ModelRunner.from_dir(engine_dir=self.engine_dir)

    def run_inference(self, prompt):
        batch_input_ids = [torch.tensor(self.tokenizer.encode(prompt), dtype=torch.int32).unsqueeze(0)]
        input_lengths = [len(input_id) for input_id in batch_input_ids]

        with torch.no_grad():
            outputs = self.runner.generate(
                batch_input_ids,
                max_new_tokens=self.max_output_len,
                max_attention_window_size=self.max_attention_window_size,
                end_id=self.end_id,
                pad_id=self.tokenizer.pad_token_id,
                temperature=1.0,
                top_k=1,
                top_p=0.0,
                num_beams=1,
                length_penalty=1.0,
                repetition_penalty=1.0,
                return_dict=True
            )
            torch.cuda.synchronize()

        output_ids = outputs['output_ids'][0][0]  # assuming we want the first output and not using beams
        generated_text = self.tokenizer.decode(output_ids[input_lengths[0]:], skip_special_tokens=True)
        return generated_text
    

if __name__ == "__main__":
    inference = LLMInference(
        max_output_len=150,
        tokenizer_dir="mistral7b_hf_tokenizer",
        engine_dir="converted",
        max_attention_window_size=4096
    )

    print("Running inference...")


    full_prompt = """
    VISUAL PROMPTS HISTORY :
    None

    TEXT:
    The fall weather in New York City was beautiful today. The light breeze prickled my skin as I sat on a wooden bench. I was in a park that was just a block away from my apartment, staring into nothing in particular. The leaves had fallen onto the ground as the trees were getting ready for winter.
    I had been trying to steal my thoughts away from what had happened the past week. A very well-respected man like Wilson Alexander didn't deserve this. He was my boss and at the same time, he was already like my father. I had worked for him as his assistant for the past three years. Never had I encountered a hard time working for him. He was always so cheery and full of life. He might have lost his temper at times but he had never failed to apologize right after he cooled down. I could never blame him though; he was, after all, the CEO of Alexander Corporation. The pressure from running a business successfully was his responsibility.
    Alexander Corporation was one of the biggest companies in the hotel industry. They had several hotels all around the world with exceptional building structures and first-class services. I had never been to any of them except for the one that was located here in New York. The place was beyond spectacular if you asked me.
    I was never the type to own or live much in luxury since my parents had left me at the age of sixteen, but I lived an ordinary life during my childhood days. We didn't have much but I was contented because I had my own perfect family. We had each other. We were happy.
    After my parents were gone, I stayed with my grandmother until I graduated from high school. As soon as graduation was done, I packed all of my things to find a job opportunity in New York. Since I was already of legal age then, I decided to support myself from then on. And now at twenty-three years old, I could afford to live on my own. I still visited my grandmother from time to time, especially during the holidays. Although I knew that I shouldn't have left her behind since she was the only family I had, I couldn't live off of my grandmother for the rest of my life.
    Wilson Alexander found me three years ago. It was around a year and a half after I left for New York City. That day, I woke up early to dress into corporate attire. I wanted a change of scene from waiting tables. So, I decided to submit applications to several companies that needed a rank and file employee.
    Alexander Corporation was my fifth stop that day. I remembered that my feet were beginning to hurt after walking in heels all morning trying to impress everyone. I also remembered smiling too much, my cheeks aching from greeting a lot of people who worked inside the building. I was exhausted.
    There was no way these people were going to hire me, I thought sourly. I was on the verge of giving up but I kept moving forward. I dragged my feet onto the tiled floor without really looking to where I was going until I bumped into someone.
    The force made me lose a step and was about to fall when I suddenly felt strong hands that held both of my arms to steady myself.
    "Oh god, I'm so sorry!" I practically yelled in panic.
    Wide-eyed, I tried composing myself and stepped back from the person that I just bumped into. I looked up and saw those golden eyes looking at me in amusement.
    I heard him chuckle and said, "Woah there, you should watch where you're going and maybe try to relax. You look disheveled."
    "I'm so sorry, sir. I wasn't paying attention to where I'm going," I said nervously.
    He then smiled warmly at me. "Well, may I ask what a pretty girl like you is doing here inside my building?"
    Did he just say that this was his building?
    I gulped. "Excuse me?"
    "What's your business here in Alexander Corporation?"
    "I-I just submitted my resume at Human Resources."
    He nodded in understanding. He then stared at me intently for a few seconds before he said, "You have no idea who I am, do you?" He paused as he waited for me to reply. When I didn't answer, he grinned at me. "Wilson Alexander, owner, and CEO of this building. It's a pleasure to meet you, Miss...?" He extended his hand to offer a handshake. I was pretty surprised at how he didn't sound arrogant when he introduced himself. I just stared at him awkwardly and then came into the realization that he wanted to shake my hand.
    I then began to stutter again. "O-oh...um..." I laughed at my silliness since I wasn't usually this nervous. "Olivia. My name's Olivia Bailey." I formally shook his hand and I knew that my hands began to feel sweaty. I had been a nervous wreck in front of the CEO of Alexander Corporation. It was not a good first impression.
    "Such a pretty name." He smiled warmly then an unexpected question escaped from his lips. "I'm headed out for lunch. Care to join me? I have a proposition that might interest you."
    We just met and he immediately wanted to have lunch together? I didn't even know this man. Even if he was interested in me, I wasn't the type to date older men. When I meant older men, like grandpa old. I wasn't going to lie; he looked good for his age. How he dressed in a well-tailored suit made him look pristine. He had hazel eyes that were wrinkling from the sides. His silvering hair was noticeable, giving away the fact that he was probably around his fifties.
    I was silent for a while, contemplating if this was a good idea. Why would he even suggest that we have lunch together? What was so special about me? I was just an ordinary woman with a job experience that was not worth bragging about.
    I heard him chuckle all of a sudden. I looked up and met his warm gaze. "If you're thinking that I'm asking you out to a lunch date then you're wrong. I'm interested in the potential that I see in you right now. I might offer you a position here in this company. Are you up for a lunch interview?"
    Was he serious?
    He was willing to give me a job that easily?
    "Really?" I couldn't contain the excitement that bubbled inside of me but I had to hold it in until after I finished lunch with Mr. Alexander. I couldn't believe my luck. I never thought that this kind of opportunity could happen to me. I was still a bit wary about all of this but it might turn my life around for the better. We would just have to see.
    He smiled at me again. "Yes. I'm not usually this informal but I have a different feeling about you. Have you decided if you're joining me for lunch?"
    I nodded silently. He then asked, "Shall we?"
    He gestured his hand for me to go first. I started to walk as he fell into step with me. I looked up and smiled at him brightly. "Thank you, Mr. Alexander."
    "Please. Call me Wilson."
    ENDTEXT

    Task: Given a lengthy text, condense its content into a concise summary, ideally within 50 words. This summary should be crafted to inspire the creation of a visual image using the Stable Diffusion model. The summary must encapsulate the main themes and elements of the text in a way that can be vividly depicted in an image. Be selective and precise in language to capture the essence of the text that translates well into a visual medium.

    Summary:
    """

    generated_text = inference.run_inference(full_prompt)
    print(generated_text)

    history = []
    text = ""
    full_prompt_template = f"""
    {history if history else ""}

    TEXT:
    {text}
    ENDTEXT

    Task: Given a lengthy text, condense its content into a concise summary, ideally within 50 words. This summary should be crafted to inspire the creation of a visual image using the Stable Diffusion model. The summary must encapsulate the main themes and elements of the text in a way that can be vividly depicted in an image. Be selective and precise in language to capture the essence of the text that translates well into a visual medium.

    Summary:
    """
