import gradio as gr
import tempfile
import os
from video_caption_generator import VideoCaptionGenerator
from typing import List, Tuple

# Initialize the generator
generator = VideoCaptionGenerator(use_huggingface=True)

class ChatState:
    """Class to maintain conversation state"""
    def __init__(self):
        self.history: List[Tuple[str, str]] = []
        self.video_context = {}
        self.video_processed = False

state = ChatState()

def process_video(video_file):
    """Handle video upload and processing"""
    global state
    state = ChatState()  # Reset state
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        tmp.write(video_file)
        video_path = tmp.name
    
    try:
        results = generator.process_video(video_path, num_frames=10)
        state.video_context = results
        state.video_processed = True
        
        initial_msg = (
            "I've analyzed your video. Here's the summary:\n\n"
            f"{results['summary']}\n\n"
            "You can ask me about:\n"
            "- Specific objects (What color is the horse?)\n"
            "- Frame details (Describe frame 3)\n"
            "- General content (What's happening?)"
        )
        
        state.history.append(("AI", initial_msg))
        
        return state.history, results['summary'], gr.update(interactive=True)
    
    except Exception as e:
        state.history.append(("Error", f"Video processing failed: {str(e)}"))
        return state.history, str(e), gr.update(interactive=False)
    finally:
        if os.path.exists(video_path):
            os.unlink(video_path)

def respond(message, chat_history):
    """Generate response to user question"""
    global state
    
    if not state.video_processed:
        chat_history.append(("AI", "Please upload a video first!"))
        return chat_history, ""
    
    try:
        chat_history.append(("You", message))
        
        answer = generator.generate_answer(
            question=message,
            context=state.video_context["summary"],
            captions=state.video_context["captions"]
        )
        
        chat_history.append(("AI", answer))
        return chat_history, ""
    
    except Exception as e:
        chat_history.append(("Error", f"Sorry, I encountered an error: {str(e)}"))
        return chat_history, ""

def create_demo():
    """Create Gradio interface"""
    with gr.Blocks(title="Video Chatbot", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🎥 Video Understanding Chatbot")
        gr.Markdown("Upload a video and chat with AI about its content")
        
        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.File(
                    label="Upload Video",
                    file_types=["video"],
                    type="binary"
                )
                upload_btn = gr.Button("Process Video", variant="primary")
                video_summary = gr.Textbox(
                    label="Video Summary",
                    interactive=False,
                    lines=6
                )
            
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    bubble_full_width=False,
                    avatar_images=(
                        None,  # User avatar (default)
                        None   # Bot avatar (default)
                    ),
                    height=500
                )
                msg = gr.Textbox(
                    label="Your question",
                    placeholder="What color is the horse? Describe frame 3...",
                    interactive=False
                )
                submit_btn = gr.Button("Send", variant="primary")
        
        # Event handlers
        upload_btn.click(
            process_video,
            inputs=video_input,
            outputs=[chatbot, video_summary, msg]
        )
        
        submit_btn.click(
            respond,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg]
        )
        
        msg.submit(
            respond,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg]
        )
        
        # Example questions
        gr.Examples(
            examples=[
                "What color is the horse?",
                "Describe what's happening in frame 3",
                "Is there a person in the video?",
                "What are the dominant colors?",
                "What is the setting of this video?"
            ],
            inputs=msg,
            label="Try these example questions:"
        )
    
    return app

if __name__ == "__main__":
    app = create_demo()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False
    )