import os
import json
import openai
import time
from PIL import Image
import base64
from tqdm import tqdm
import re


api_key = ""


input_root = ""
output_root = ""



def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def extract_all_qa_pairs(raw_text):
    qa_pairs = []

    mcq_pattern = re.findall(
        r"### Question \d+: (.*?)\n(.*?)\n\*\*Correct Answer: (.*?)\*\*",
        raw_text,
        re.DOTALL
    )
    for q_text, options_block, answer_text in mcq_pattern:
        question = q_text.strip()
        options = options_block.strip().split("\n")
        formatted_q = f"{question}\n" + "\n".join(opt.strip() for opt in options)
        qa_pairs.append({
            "question": formatted_q,
            "answer": answer_text.strip()
        })

    
    open_qas = re.findall(r"\*\*Question:\*\* (.*?)\n\*\*Answer:\*\* (.*?)\n", raw_text, re.DOTALL)
    for q, a in open_qas:
        qa_pairs.append({
            "question": q.strip(),
            "answer": a.strip()
        })

    return qa_pairs



def deduplicate_qa(qa_list):
    combined = {}
    for qa in qa_list:
        q = qa["question"]
        a = qa["answer"]
        if q in combined:
            existing_answers = combined[q].split("\n")
            new_answers = a.split("\n")
            merged = list(dict.fromkeys(existing_answers + new_answers))
            combined[q] = "\n".join(merged)
        else:
            combined[q] = a
    return [{"question": q, "answer": a} for q, a in combined.items()]

PROMPT = """Please analyze the following 10 sequential images extracted from a video. 
Each image includes a bounding box that tracks the same underwater observation target over time.
The main observation target is the object within the bounding box.

###[REQUEST]:
You have a single video featuring a marine species. Please create questions based on the following two directions: Multiple-choice section:4 Q-A pairs; Open-ended section: 16 Q-A pairs
totaling 20 question-and-answer pairs.

Multiple-choice section (4 Q-A pairs).
Craft four questions, each followed by exactly three answer options (A/B/C) and indicate the correct choice.
Important: Do not recycle a template; each question should read naturally and distinctively.
Here are the four directions for the four questions, one question per direction:

Direction1-Identification. Ask in your own words what species the images most likely portray.
Direction2-Next Action Prediction. Phrase in any style a question about what the organism will probably do next.
Direction3-Movement Direction Prediction. Using fresh wording, enquire where the animal is likely headed.
Direction4-Clip Summary. Formulate a comparative question that lets the reader pick the best high-level description of the entire video.

Open-ended section (16 Q-A pairs).
Create one question and one answer for each of the 16 aspects listed below.
Vary your question styles (direct, rhetorical, hypothetical, curiosity-driven, elaboration requests, etc.).
Diversify wording and sentence structure across both questions and answers.
When needed, infer plausible details from established biological knowledge (e.g., typical diet or habitat of the species).
In the scene of the image, if multiple different species appear in the background, please use non-questioning features to refer to the target when generating questions. If it is unavoidable to use questioning features, then use non-currently-questioned features to refer to the target (e.g., when asking about size, use color or trajectory as a reference).

**Please strictly follow the format shown in [EXAMPLE].And do not include any bounding box information in either the questions or the answers.**

Requirements:
###Biological Characteristics
Size: Estimated length with confidence range.
Color: Dominant colors, patterns, and any bioluminescence.
Habitat: Specific ecosystem (barrier reef, seagrass bed, pelagic zone, etc.). Include inferences about diet and behavior based on the species.
Diet: Feeding type (carnivore, herbivore) and examples of typical prey.
Behavior: Social patterns (schooling/solitary) and hunting or foraging strategies.

###Observed Target Characteristics
Pose:Screen-space trajectory, swim angle, and relative speed (fast / moderate / slow)
Action: Main behavior observed, plus a timeline (e.g., how it swims at the beginning and where it moves throughout the video).
Expression: Inferred emotional or health indicators based on eyes, mouth, gills, etc.


###Illustrated Environmental Features
Region_type: Sandy, rocky, coral, or other substrate.
Light_conditions: Overall lighting and how it changes over time.
Visibility: Overall clarity of the water and variations over the video’s duration.
Environment: Overall setting description and notable scene changes throughout the video.

###Physicochemical Water Properties
Water_type: Typical habitat type (saltwater, brackish, etc.) inferred from the species’ known range.
Salinity: Tolerance range of the species.
Current_strength: Observations about water current and potential adaptations.
Temperature: Optimal temperature range and any noted signs of temperature impact.


###[EXAMPLE]:

Question: What species is most likely depicted by the fish with blue and yellow coloring in the middle of the scene?
A) Surgeonfish
B) Butterflyfish
C) Angelfish
Correct Answer: A) Surgeonfish

Question: Considering the fish with vibrant blue and yellow hues, what is it most likely to do next in the video?
A) Continue swimming forward
B) Swim toward the coral structure
C) Change direction rapidly to avoid another fish
Correct Answer: B) Swim toward the coral structure

Question: Based on the movement of the fish with blue and yellow stripes, where is it likely headed?
A) Deeper into the coral reef
B) Toward the surface
C) Moving towards open water away from the reef
Correct Answer: A) Deeper into the coral reef

Question: Which of the following best describes the fish’s behavior throughout the video, based on its interaction with the surrounding environment?
A) The fish swims through a vibrant reef, interacting with various other species.
B) The fish remains stationary, hiding within the reef structure.
C) The fish interacts only with others of its own kind, moving in groups.
Correct Answer: A) The fish swims through a vibrant reef, interacting with various other species.

Open-Ended Section
Biological Characteristics

Question: How large do you estimate the fish with the dominant blue and yellow coloration to be, based on its size relative to the surrounding fish?
Answer: The fish appears to be around 25-30 cm in length, a typical size for a Surgeonfish.

Question: What are the primary colors of the fish in the center of the scene, distinguished by its blue and yellow pattern?
Answer: The fish shows vibrant blue and yellow hues across its body, common for a Surgeonfish.

Question: What type of habitat is the fish with blue and yellow coloring likely living in, based on the surrounding environment in the scene?
Answer: The fish likely inhabits a coral reef ecosystem, providing shelter and food sources such as algae.

Question: What is the diet of the fish with blue and yellow colors, and what does it typically consume?
Answer: The fish is herbivorous, typically feeding on algae and other plant materials found within the reef.

Question: How would you describe the social behavior and feeding pattern of the fish, given its movement in the video?
Answer: The fish appears to be solitary, moving along the reef and foraging for algae, typical of many Surgeonfish species.

Observed Target Characteristics

Question: How is the fish positioned while swimming, and what can you infer about its swimming speed based on the image?
Answer: The fish has a horizontal posture and swims at a moderate pace, typical of foraging behaviors.

Question: What is the fish with blue and yellow markings doing in the video, and how does its behavior evolve?
Answer: The fish is swimming steadily, grazing on algae, then continues its path through the reef, often pausing to feed.

Question: What can we infer about the fish’s health or emotional state from its body language in the video?
Answer: The fish seems healthy and calm, as it swims with fluidity and engages in natural foraging behavior.

Illustrated Environmental Features

Question: What type of substrate is the fish with blue and yellow coloration inhabiting in the scene?
Answer: The fish is swimming in a coral reef, characterized by rocky surfaces and vibrant coral formations.

Question: How does the lighting in the video affect the appearance of the fish with blue and yellow coloring?
Answer: The lighting is bright, making the colors of the fish more vivid and allowing clear visibility of its vibrant body.

Question: How would you describe the water visibility, and how does it impact the fish's appearance?
Answer: The water is crystal clear, allowing the fish's details to be sharply visible, with no distortion from murky water.

Question: What is the overall environment in the video, and how does it influence the fish’s behavior?
Answer: The fish is swimming in a dynamic coral reef with various species around, providing a rich environment for foraging and movement.

Physicochemical Water Properties

Question: What type of water is the fish with blue and yellow markings likely inhabiting?
Answer: The fish is most likely in saltwater, typical for a coral reef ecosystem where this species is found.

Question: How well does the fish tolerate different salinity levels in its habitat?
Answer: The fish is adapted to the stable salinity levels found in coral reef waters, which are typically high.

Question: Based on the fish's movements, how would you rate the strength of the current in the water?
Answer: The current is mild, as the fish swims at a steady pace without being pushed by strong currents.

Question: What is the optimal temperature range for this species, and is the water temperature in the video suitable for it?
Answer: The fish thrives in water temperatures between 24°C and 28°C, which appears to be the case in the clear, bright water of the video.

"""
# Please answer the following questions about the fish in the video that exhibits a vibrant mix of blue and yellow and swims steadily through the central area of the coral reef.
client = openai.OpenAI(api_key=api_key) 

folders = sorted([f for f in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, f))])

start_idx = 104
for folder_idx, folder in enumerate(tqdm(folders[start_idx-1:], desc="Processing Folders"),start=start_idx):
    folder_path = os.path.join(input_root, folder)


    image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))])

    mes =[
            {"role": "system", "content": "You are an expert in marine biology and image analysis."},
            {"role": "user", "content": 
                [
                    {
                        "type": "text",
                        "text": PROMPT
                    }
                ]
            },
        ]


    for img_idx, image_file in enumerate(image_files):
        image_path = os.path.join(folder_path, image_file)  
        base64_image = encode_image(image_path)
        image_data = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/{'jpg'};base64,{base64_image}"
                }
        }
        mes[1]['content'].append(image_data)

    response = client.chat.completions.create(
            model="gpt-4o",
            messages=mes,
            max_tokens=4096
        )
    
    raw_output = response.choices[0].message.content


    raw_output_txt_filename = os.path.join(output_root, f"{folder}_response.txt")
    with open(raw_output_txt_filename, "w", encoding="utf-8") as f:
        f.write(raw_output)

    repairs = extract_all_qa_pairs(raw_output)
    repairs = deduplicate_qa(repairs)
    # print(response.choices[0].message.content)

    qa_json_filename = os.path.join(output_root, f"{folder}_qa.json")
    with open(qa_json_filename, "w", encoding="utf-8") as f:
        json.dump(repairs, f, indent=4, ensure_ascii=False)


    time.sleep(5)

print("Work finish!")
