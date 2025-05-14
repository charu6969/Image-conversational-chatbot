import os
import torch
import numpy as np
from PIL import Image
import cv2
import time
import random
import tempfile
from typing import Dict, Union, List, Optional

# Check if transformers is available
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers library not fully available. Some features may be limited.")

class VideoCaptionGenerator:
    def __init__(self, use_huggingface=True, api_key=None):
        """
        Initialize the VideoCaptionGenerator with either Hugging Face models or API key.
        """
        self.use_huggingface = use_huggingface
        self.api_key = api_key
        
        if use_huggingface:
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("The transformers library is required when use_huggingface=True")
                
            print("Loading models...")
            try:
                # Image captioning model
                self.caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                self.caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
                
                # Text generation models
                self.question_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
                self.question_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
                self.nlp = pipeline("text2text-generation", model="facebook/bart-large-cnn")
                
                # QA model
                self.qa_pipeline = pipeline(
                    "question-answering",
                    model="deepset/roberta-base-squad2",
                    tokenizer="deepset/roberta-base-squad2"
                )
                
                print("All models loaded successfully.")
            except Exception as e:
                print(f"Error loading models: {e}")
                raise

    def extract_frames(self, video_path, num_frames=10):
        """Extract frames from video file."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            raise ValueError(f"Could not read frames from video: {video_path}")
        
        frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
        frames = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
        
        cap.release()
        return frames

    def generate_caption_for_frame(self, frame):
        """Generate caption for a single frame."""
        if self.use_huggingface:
            try:
                inputs = self.caption_processor(frame, return_tensors="pt")
                output = self.caption_model.generate(**inputs, max_length=30)
                return self.caption_processor.decode(output[0], skip_special_tokens=True)
            except Exception as e:
                print(f"Error generating caption: {e}")
                return "Could not caption this frame"

    def generate_captions(self, video_path, num_frames=10):
        """Generate captions for all frames in video."""
        frames = self.extract_frames(video_path, num_frames)
        return [self.generate_caption_for_frame(frame) for frame in frames]

    def summarize_video_content(self, captions):
        """Generate summary from frame captions."""
        combined_captions = " ".join(captions)
        
        if self.use_huggingface:
            try:
                inputs = self.question_tokenizer(combined_captions, return_tensors="pt", 
                                              max_length=1024, truncation=True)
                summary_ids = self.question_model.generate(
                    inputs.input_ids,
                    max_length=150,
                    min_length=40,
                    length_penalty=2.0,
                    num_beams=4,
                    early_stopping=True
                )
                return self.question_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            except Exception as e:
                print(f"Error generating summary: {e}")
                return combined_captions[:200] + "..."

    def generate_followup_questions(self, captions, summary, num_questions=3):
        """Generate follow-up questions about video."""
        if self.use_huggingface:
            try:
                context = f"Content: {summary}\nDetails: {' '.join(captions[:3])}"
                questions = []
                
                for _ in range(num_questions):
                    result = self.nlp(
                        f"Generate one question about: {context}",
                        max_length=50,
                        num_return_sequences=1,
                        temperature=0.7
                    )[0]['generated_text']
                    questions.append(result.strip())
                
                return questions
            except Exception as e:
                print(f"Error generating questions: {e}")
                return ["What is happening in the video?"] * num_questions

    def generate_answer(self, question: str, context: str, captions: List[str]) -> str:
        """Generate specific answer to video-related question."""
        if self.use_huggingface:
            try:
                lower_question = question.lower()
                
                # Enhanced color detection
                if any(color_word in lower_question for color_word in ["color", "colour"]):
                    return self._analyze_colors(captions, specific_object=self._extract_object(lower_question))
                
                # Opinion questions
                elif "opinion" in lower_question:
                    return "As an AI, I don't have personal opinions. Historically, horses have been used for transportation, work, and recreation."
                
                # Frame-specific questions
                elif any(word in lower_question for word in ["frame", "specific", "detail"]):
                    return self._answer_frame_question(lower_question, captions)
                
                # Default QA approach
                else:
                    qa_context = self._build_qa_context(context, captions)
                    qa_result = self.qa_pipeline({
                        'question': question,
                        'context': qa_context
                    })
                    
                    if qa_result['score'] > 0.4:
                        return qa_result['answer']
                    
                    # Fallback to generation
                    return self._generate_fallback_answer(question, qa_context)
            
            except Exception as e:
                print(f"Error generating answer: {e}")
                return "I couldn't analyze that specific aspect. Please try another question."

    def _analyze_colors(self, captions: List[str], specific_object: str = None) -> str:
        """Analyze frame captions to determine colors."""
        color_map = {
            'brown': ['brown', 'tan', 'chestnut'],
            'black': ['black', 'dark'],
            'white': ['white', 'light'],
            'gray': ['gray', 'grey'],
            'green': ['green', 'grass', 'leaf', 'field'],
            'blue': ['blue', 'sky'],
            'red': ['red', 'orange'],
            'yellow': ['yellow', 'gold']
        }
        
        detected_colors = set()
        
        for caption in captions[:5]:  # Check first 5 frames
            lower_caption = caption.lower()
            
            if specific_object and specific_object in lower_caption:
                for color, keywords in color_map.items():
                    if any(keyword in lower_caption for keyword in keywords):
                        detected_colors.add(color)
            else:
                for color, keywords in color_map.items():
                    if any(keyword in lower_caption for keyword in keywords):
                        detected_colors.add(color)
        
        if detected_colors:
            if specific_object:
                return f"The {specific_object} appears to be {', '.join(detected_colors)}."
            return f"The dominant colors are: {', '.join(detected_colors)}"
        
        return "I couldn't determine the colors from the video."

    def _extract_object(self, question: str) -> Optional[str]:
        """Extract object of interest from color question."""
        objects = ['horse', 'sky', 'grass', 'field', 'person', 'clothing', 'shirt']
        for obj in objects:
            if obj in question:
                return obj
        return None

    def _answer_frame_question(self, question: str, captions: List[str]) -> str:
        """Handle frame-specific questions."""
        try:
            frame_num = None
            words = question.split()
            for i, word in enumerate(words):
                if word.isdigit():
                    frame_num = int(word)
                    break
            
            if frame_num and 1 <= frame_num <= len(captions):
                return f"Frame {frame_num} shows: {captions[frame_num-1]}"
            return "Please specify a valid frame number to examine."
        except:
            return "I couldn't understand which frame you're asking about."

    def _build_qa_context(self, summary: str, captions: List[str]) -> str:
        """Build comprehensive context for QA."""
        return f"""Video Summary: {summary}
        
        Frame Details:
        {chr(10).join(f"- {caption}" for i, caption in enumerate(captions[:3]))}
        
        Additional Information:
        - Common objects: people, animals, vehicles, buildings
        - Typical settings: indoor, outdoor, urban, rural
        - Time indicators: day, night, sunrise, sunset
        """

    def _generate_fallback_answer(self, question: str, context: str) -> str:
        """Generate answer when QA pipeline isn't confident."""
        prompt = f"""Context: {context}
        Question: {question}
        Provide a concise answer:"""
        
        generated = self.nlp(
            prompt,
            max_length=100,
            num_return_sequences=1,
            temperature=0.7
        )[0]['generated_text']
        
        return generated.split(":")[-1].strip()

    def process_video(self, video_path, num_frames=10, num_questions=3):
        """Full video processing pipeline."""
        captions = self.generate_captions(video_path, num_frames)
        summary = self.summarize_video_content(captions)
        questions = self.generate_followup_questions(captions, summary, num_questions)
        
        return {
            "captions": captions,
            "summary": summary,
            "questions": questions
        }