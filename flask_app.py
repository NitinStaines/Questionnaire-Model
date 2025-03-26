from flask import Flask, request, jsonify
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
import time
import re
import textwrap
from transformers import BartTokenizer, BartForConditionalGeneration
import numpy as np
from sense2vec import Sense2Vec
s2v = Sense2Vec().from_disk('../s2v_old')
from sentence_transformers import SentenceTransformer
sentence_transformer_model = SentenceTransformer('msmarco-distilbert-base-v3')
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from similarity.normalized_levenshtein import NormalizedLevenshtein
from collections import OrderedDict
from sklearn.metrics.pairwise import cosine_similarity
normalized_levenshtein = NormalizedLevenshtein()

def filter_same_sense_words(original,wordlist):
  filtered_words=[]
  base_sense =original.split('|')[1]
  print (base_sense)
  for eachword in wordlist:
    if eachword[0].split('|')[1] == base_sense:
      filtered_words.append(eachword[0].split('|')[0].replace("_", " ").title().strip())
  return filtered_words

def get_highest_similarity_score(wordlist,wrd):
  score=[]
  for each in wordlist:
    score.append(normalized_levenshtein.similarity(each.lower(),wrd.lower()))
  return max(score)

def sense2vec_get_words(word,s2v,topn,question):
    output = []
    #print ("word ",word)
    try:
      sense = s2v.get_best_sense(word, senses= ["NOUN", "PERSON","PRODUCT","LOC","ORG","EVENT","NORP","WORK OF ART","FAC","GPE","NUM","FACILITY"])
      most_similar = s2v.most_similar(sense, n=topn)
      # print (most_similar)
      output = filter_same_sense_words(sense,most_similar)
      #print ("Similar ",output)
    except:
      output =[]

    threshold = 0.6
    final=[word]
    checklist =question.split()
    for x in output:
      if get_highest_similarity_score(final,x)<threshold and x not in final and x not in checklist:
        final.append(x)

    return final[1:]

def mmr(doc_embedding, word_embeddings, words, top_n, lambda_param):

    # Extract similarity within words, and between words and the document
    word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding)
    word_similarity = cosine_similarity(word_embeddings)

    # Initialize candidates and already choose best keyword/keyphrase
    keywords_idx = [np.argmax(word_doc_similarity)]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    for _ in range(top_n - 1):
        # Extract similarities within candidates and
        # between candidates and selected keywords/phrases
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

        # Calculate MMR
        mmr = (lambda_param) * candidate_similarities - (1-lambda_param) * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        # Update keywords & candidates
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    return [words[idx] for idx in keywords_idx]


def get_distractors_wordnet(word):
    distractors=[]
    try:
      syn = wn.synsets(word,'n')[0]

      word= word.lower()
      orig_word = word
      if len(word.split())>0:
          word = word.replace(" ","_")
      hypernym = syn.hypernyms()
      if len(hypernym) == 0:
          return distractors
      for item in hypernym[0].hyponyms():
          name = item.lemmas()[0].name()
          #print ("name ",name, " word",orig_word)
          if name == orig_word:
              continue
          name = name.replace("_"," ")
          name = " ".join(w.capitalize() for w in name.split())
          if name is not None and name not in distractors:
              distractors.append(name)
    except:
      print ("Wordnet distractors not found")
    return distractors

def get_distractors (word,origsentence,sense2vecmodel,sentencemodel,top_n,lambdaval):
  distractors = sense2vec_get_words(word,sense2vecmodel,top_n,origsentence)
  #print ("distractors ",distractors)
  if len(distractors) ==0:
    return distractors
  distractors_new = [word.capitalize()]
  distractors_new.extend(distractors)
  # print ("distractors_new .. ",distractors_new)

  embedding_sentence = origsentence+ " "+word.capitalize()
  # embedding_sentence = word
  keyword_embedding = sentencemodel.encode([embedding_sentence])
  distractor_embeddings = sentencemodel.encode(distractors_new)

  # filtered_keywords = mmr(keyword_embedding, distractor_embeddings,distractors,4,0.7)
  max_keywords = min(len(distractors_new),5)
  filtered_keywords = mmr(keyword_embedding, distractor_embeddings,distractors_new,max_keywords,lambdaval)
  # filtered_keywords = filtered_keywords[1:]
  final = [word.capitalize()]
  for wrd in filtered_keywords:
    if wrd.lower() !=word.lower():
      final.append(wrd.capitalize())
  final = final[1:4]
  return final

app = Flask(__name__)

from flask_cors import CORS
app = Flask(__name__)
CORS(app)

# Check if GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print("There are %d GPU(s) available." % torch.cuda.device_count())
    print("We will use the GPU:", torch.cuda.get_device_name(0))

# Load the model and tokenizer, and move model to the selected device
model = T5ForConditionalGeneration.from_pretrained(r'D:\College\SDP_Project\backend\model\model').to(device)
tokenizer = T5Tokenizer.from_pretrained(r'D:\College\SDP_Project\backend\model\tokenizer')

def generate_qa_pairs(context, num_questions=1):
    try:
        # Prepare the input for the model
        input_text = f"context: {context}"
        input_ids = tokenizer(input_text, return_tensors='pt', max_length=512, truncation=True).input_ids.to(device)

        # Generate questions
        question_outputs = model.generate(
            input_ids,
            max_length=50,
            num_return_sequences=num_questions,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            temperature=1.0
        )
        questions = tokenizer.batch_decode(question_outputs, skip_special_tokens=True)

        # Prepare to generate answers for each question
        qa_pairs = []
        for question in questions:
            answer_input = f"question: {question} context: {context}"
            answer_ids = tokenizer(answer_input, return_tensors='pt', max_length=512, truncation=True).input_ids.to(device)

            # Generate answers
            answer_output = model.generate(answer_ids, max_length=50)
            answer = tokenizer.decode(answer_output[0], skip_special_tokens=True)

            qa_pairs.append({"question": question, "answer": answer})

        return qa_pairs
    except Exception as e:
        print("Error generating QA pairs:", e)
        return []

def clean_article_content(raw_content):
    """
    Cleans the scraped article content to remove unnecessary information.

    Args:
        raw_content (str): The raw content extracted from the article.

    Returns:
        str: The cleaned content with unnecessary information removed.
    """

    # Define a list of phrases and patterns to remove
    unnecessary_phrases = [
        r'(?i)\bADVERTISEMENT\b',  # Remove 'ADVERTISEMENT'
        r'(?i)(Pause|Unmute|Fullscreen)\s*:\s*.*?(\n|$)',  # Remove 'Pause', 'Unmute', and 'Fullscreen'
        r'(?i)Also Read.*?(\n|$)',  # Remove 'Also Read' sections
        r'STORIES YOU MAY LIKE.*?(\n|$)',  # Remove 'STORIES YOU MAY LIKE'
        r'(?i)Explained.*?(\n|$)',  # Remove 'Explained' sections
        r'\[.*?\]',  # Remove any text in brackets
        r'\(.*?\)',  # Remove any text in parentheses
        r'(?i)\b(Watch|Listen)\b.*?(\n|$)',  # Remove 'Watch', 'Listen', etc.
        r'(?i)\bRead more\b.*?(\n|$)',  # Remove 'Read more'
        r'\b(PTI|ANI|Reuters|The Indian Express)\b',  # Remove agency names
        # r'\bIn his statement\b.*?(\n|$)',  # Remove introductory phrases
    ]

    # Remove unnecessary phrases using the defined patterns
    for pattern in unnecessary_phrases:
        raw_content = re.sub(pattern, '', raw_content)

    # Remove any lines that contain excessive whitespace or are too short (less than a certain length)
    cleaned_content = '\n'.join(line for line in raw_content.splitlines() if len(line.strip()) > 40)

    # Strip leading and trailing whitespace from the final cleaned content
    cleaned_content = cleaned_content.strip()

    return cleaned_content

def extract_article_content(driver):
    try:
        # Try locating the article content using common CSS selectors
        # You should update this selector based on the actual webpage structure
        article = driver.find_element(By.CSS_SELECTOR, "div.article-body")  # Example: Update to match the site's structure
        #print("Article found using div.article-body")

    except NoSuchElementException:
        try:
            # If the first method fails, try finding the article by a common tag like <article>
            article = driver.find_element(By.TAG_NAME, "article")
            #print("Article found using <article> tag")

        except NoSuchElementException:
            try:
                # Try locating the article using another possible class or ID
                article = driver.find_element(By.CSS_SELECTOR, "div[class*='content']")  # Fallback example
                #print("Article found using div[class*='content']")

            except NoSuchElementException:
                print("Could not find the article element")
                return None

    return article.text

# Set up the web driver (Chrome in this case)
def setup_driver():
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')  # Run in headless mode (no GUI)
    options.add_argument('--disable-gpu')  # Disable GPU acceleration
    options.add_argument('--no-sandbox')  # Bypass OS security model (for Linux users)
    driver = webdriver.Chrome(options=options)
    return driver

# Main function to load the webpage and extract the article
def get_article_content(url):
    driver = setup_driver()
    try:
        # Open the page
        driver.get(url)

        # Wait for the page to load completely (you can adjust the wait time)
        time.sleep(5)

        # Extract the article content
        content = extract_article_content(driver)
        print("This is the content--->",content)
        if content:
            print("Article content extracted successfully")
            return content
        else:
            print("Failed to extract the article content.")

    finally:
        driver.quit()
model_path = "fine-tuned-bart"

def split_text(text, max_length):
    """Splits text into chunks of max_length tokens."""
    tokenizer = BartTokenizer.from_pretrained(model_path)
    tokens = tokenizer.encode(text, return_tensors="pt")[0]
    chunks = [tokens[i:i+max_length] for i in range(0, len(tokens), max_length)]
    return chunks

def summarize_chunk(chunk, model, tokenizer):
    """Generates a summary for a single chunk of text tokens."""
    inputs = chunk.unsqueeze(0)
    summary_ids = model.generate(inputs, max_length=512, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def further_summarize(text, target_length, model, tokenizer):
    """Generates a summary for the text to achieve the target length."""
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=target_length*2, min_length=target_length, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def text_summarizer(text):
    # Load the fine-tuned BART model and tokenizer
    model = BartForConditionalGeneration.from_pretrained(model_path)
    tokenizer = BartTokenizer.from_pretrained(model_path)

    # Split text into manageable chunks
    max_length = 1024  # Maximum number of tokens for BART
    chunks = split_text(text, max_length)

    # Summarize each chunk
    summaries = [summarize_chunk(chunk, model, tokenizer) for chunk in chunks]

    # Combine summaries
    combined_summary = " ".join(summaries)

    # Determine the length of the original text
    original_length = len(tokenizer.encode(text, return_tensors="pt")[0])
    target_summary_length = original_length // 2  # Target length is half of the original

    # Ensure the final summary meets the target length
    final_summary = combined_summary
    while len(tokenizer.encode(final_summary, return_tensors="pt")[0]) > target_summary_length:
        final_summary = further_summarize(final_summary, target_summary_length, model, tokenizer)



    # Decode and return the final formatted summary
    formatted_summary = "\n".join(textwrap.wrap(final_summary, width=80))  # Adjust width as needed
    return formatted_summary

from flask import Flask, request, jsonify
# Ensure all necessary imports are included
# ...

import spacy

# Load spaCy model for POS tagging
nlp = spacy.load("en_core_web_sm")

def identify_noun(answer):
    doc = nlp(answer)
    # Find the first noun in the answer
    for token in doc:
        if token.pos_ == "NOUN":
            return token.text
    return None  # If no noun found, return None

@app.route('/scrape', methods=['POST'])
def scrape():
    # Receive JSON data from frontend
    data = request.get_json()
    url = data.get("url")
    
    print("Data received by Flask:", url)
    
    # Check if URL is provided
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    # Scrape the article content
    article_content = get_article_content(url)
    print("\nArticle Content:", article_content, "\n")

    # Clean the article content
    cleaned_content = clean_article_content(article_content)
    cleaned_content = text_summarizer(cleaned_content)
    # Generate QA pairs
    qa_pairs = generate_qa_pairs(cleaned_content, num_questions=5)
    distractor_results = []

    for qa in qa_pairs:
        question = ""
        answer = ""
        
        # Check if the question contains "answer:" and split accordingly
        if "answer:" in qa['question']:
            question_part, answer_part = qa['question'].split("answer:")
            question = question_part.replace("question:", "").strip()
            answer = answer_part.strip()
        else:
            question = qa['question'].replace("question:", "").strip()
            answer = qa['answer'].strip()

        print("Processed Question:", question)
        print("Processed Answer:", answer)

        # Step 1: Identify the noun in the answer using POS tagging
        noun = identify_noun(answer)
        
        if noun:
            print("Identified Noun:", noun)
            # Step 2: Generate distractors based on the noun
            distractors = get_distractors(noun, question, s2v, sentence_transformer_model, 40, 0.2)

            # If the answer has multiple words, append them to each distractor
            if len(answer.split()) > 1:
                distractors = [f"{dist} {' '.join(answer.split()[1:])}" for dist in distractors]
            
            print("Distractors:", distractors)
        else:
            print("No noun identified, generating distractors based on the first word of the answer.")
            # Fallback to generating distractors based on the first word of the answer
            distractors = get_distractors(answer.split()[0], question, s2v, sentence_transformer_model, 40, 0.2)
        
        # Determine question type
        question_type = "mcq" if len(answer.split()) <= 2 else "fill"
        
        # Add to results
        distractor_results.append({
            "question": question,
            "answer": answer,
            "distractors": distractors,
            "type": question_type
        })

    # Return the QA pairs and distractors in a consistent structure
    return jsonify({'results': distractor_results})


@app.route('/generate', methods=['POST'])
def predict():
    # Receive JSON data from frontend
    data = request.get_json()
    context = data.get("context")
    context = text_summarizer(context)
    if not context:
        return jsonify({"error": "No context provided"}), 400

    # Generate QA pairs
    qa_pairs = generate_qa_pairs(context, num_questions=5)
    distractor_results = []

    for qa in qa_pairs:
        question = ""
        answer = ""
        
        # Check if the question contains "answer:" and split accordingly
        if "answer:" in qa['question']:
            question_part, answer_part = qa['question'].split("answer:")
            question = question_part.replace("question:", "").strip()
            answer = answer_part.strip()
        else:
            question = qa['question'].replace("question:", "").strip()
            answer = qa['answer'].strip()

        print("Processed Question:", question)
        print("Processed Answer:", answer)

        # Step 1: Identify the noun in the answer using POS tagging
        noun = identify_noun(answer)
        
        if noun:
            print("Identified Noun:", noun)
            # Step 2: Generate distractors based on the noun
            distractors = get_distractors(noun, question, s2v, sentence_transformer_model, 40, 0.2)

            # If the answer has multiple words, append them to each distractor
            if len(answer.split()) > 1:
                distractors = [f"{dist} {' '.join(answer.split()[1:])}" for dist in distractors]
            
            print("Distractors:", distractors)
        else:
            print("No noun identified, generating distractors based on the first word of the answer.")
            # Fallback to generating distractors based on the first word of the answer
            distractors = get_distractors(answer.split()[0], question, s2v, sentence_transformer_model, 40, 0.2)
        
        # Determine question type
        question_type = "mcq" if len(answer.split()) <= 2 else "fill"
        
        # Add to results
        distractor_results.append({
            "question": question,
            "answer": answer,
            "distractors": distractors,
            "type": question_type
        })

    # Return the QA pairs and distractors in a consistent structure
    return jsonify({'results': distractor_results})



import spacy

# Load spaCy model for POS tagging
nlp = spacy.load("en_core_web_sm")

def identify_noun(answer):
    doc = nlp(answer)
    # Find the first noun in the answer
    for token in doc:
        if token.pos_ == "NOUN":
            return token.text
    return None  # If no noun found, return None





@app.route('/submit-answer', methods=['POST'])

def correct_answer():
    
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
    model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")
    data = request.get_json()
    question = data.get("question")
    answer = data.get("answer")
    correct_answer = data.get("correctAnswer")

    # Encode input sentences for NLI model
    inputs = tokenizer(answer, correct_answer, return_tensors="pt", truncation=True, padding=True)

    # Get model predictions
    outputs = model(**inputs)
    logits = outputs.logits
    probs = F.softmax(logits, dim=-1)

    # Extract the entailment probability
    entailment_prob = probs[0][2].item()

    print(f"Entailment: {entailment_prob:.4f}")

    # Return the entailment probability to the frontend
    return jsonify({"entailment_score": entailment_prob})

if __name__ == '__main__':
    app.run(debug=True)