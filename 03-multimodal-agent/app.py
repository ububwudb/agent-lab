"""多模态 Agent 的界面（Gradio）

三个标签页：
1. 语音对话 —— 说话 → 识别 → Agent 思考（会调工具）→ 语音回答
2. 摄像头/图片 —— 拍照或上传 → 视觉模型看图回答 → 可以朗读
3. 文字对话 —— 直接打字，方便你调试和观察 Agent 每一步

启动：python app.py
"""
import gradio as gr

from agent import chat, clear_memory
from asr import transcribe
from tts import speak, VOICES
from vision import ask_vl, grab_frame, save_frame


def voice_turn(audio_path, history, engine):
    """一轮语音对话：音频 -> 文字 -> Agent -> 语音"""
    if not audio_path:
        return history, None, "（没有收到录音）"
    user_text = transcribe(audio_path)
    if user_text.startswith("[ASR"):
        return history, None, user_text
    reply = chat(user_text, thread_id="voice")
    voice_file = speak(reply, engine=engine)
    history = history + [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": reply},
    ]
    return history, voice_file, user_text


def vision_turn(image, question):
    """看一张图（摄像头拍的或上传的）并回答问题"""
    if image is None:
        return "先拍一张照片，或者上传一张图片。", None
    q = (question or "").strip() or "请用一句话描述你看到了什么"
    answer = ask_vl(image, q)
    return answer, None


def read_aloud(text, engine):
    if not text or text.startswith("先拍"):
        return None
    return speak(text, engine=engine)


def text_turn(message, history):
    if not message.strip():
        return history, ""
    reply = chat(message, thread_id="text")
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": reply},
    ]
    return history, ""


def shoot():
    """直接抓一帧摄像头画面显示在界面上"""
    frame = grab_frame()
    if frame is None:
        return None, "打不开摄像头：可能被占用了，或者没有摄像头权限"
    save_frame(frame, "last_camera.jpg")
    return frame, "已抓到画面，问它点什么吧"


with gr.Blocks(title="多模态 Agent") as demo:
    gr.Markdown("""
    # 多模态 Agent
    能听（麦克风）· 会说（语音合成）· 会看（摄像头/屏幕）· 会用工具（天气/计算/笔记/开网页）
    """)

    with gr.Tabs():
        # ---------- 1. 语音对话 ----------
        with gr.TabItem("🎙️ 语音对话"):
            engine = gr.Radio(choices=["edge", "api"], value="edge",
                              label="语音合成引擎",
                              info="edge=免费本地方案（推荐），api=硅基流动 CosyVoice2（更自然，耗额度）")
            chatbot = gr.Chatbot(height=380, label="对话记录")
            heard = gr.Textbox(label="识别到的你说的话", interactive=False)
            with gr.Row():
                mic = gr.Audio(sources=["microphone"], type="filepath",
                               label="按住说话，说完点停止", waveform_options={"show_recording_waveform": True})
                out_audio = gr.Audio(label="助手的回答", autoplay=True)
            mic.stop_recording(voice_turn, [mic, chatbot, engine], [chatbot, out_audio, heard])
            gr.Button("清空语音记忆").click(
                lambda: ([], None, clear_memory("voice") or "已清空"), None, [chatbot, out_audio, heard])

        # ---------- 2. 摄像头 / 图片 ----------
        with gr.TabItem("📷 摄像头看世界"):
            with gr.Row():
                img = gr.Image(sources=["webcam", "upload"], type="numpy",
                               label="拍照或上传图片", height=320)
                v_engine = gr.Radio(choices=["edge", "api"], value="edge", label="朗读引擎")
            v_question = gr.Textbox(value="请用一句话描述你看到了什么",
                                    label="你想问这张图什么")
            v_answer = gr.Textbox(label="视觉模型的回答", lines=4)
            v_audio = gr.Audio(label="朗读回答", autoplay=True)
            with gr.Row():
                gr.Button("拍照（直接抓摄像头）").click(shoot, None, [img, v_answer])
                gr.Button("问它").click(vision_turn, [img, v_question], [v_answer, v_audio])
                gr.Button("朗读回答").click(read_aloud, [v_answer, v_engine], [v_audio])

        # ---------- 3. 文字对话 ----------
        with gr.TabItem("💬 文字对话（调试用）"):
            t_chatbot = gr.Chatbot(height=420, label="对话记录")
            t_input = gr.Textbox(label="输入一句话", placeholder="例如：沈阳和北京温差多少 / 帮我看看摄像头里有什么")
            t_send = gr.Button("发送")
            t_clear = gr.Button("清空记忆")
            t_send.click(text_turn, [t_input, t_chatbot], [t_chatbot, t_input])
            t_input.submit(text_turn, [t_input, t_chatbot], [t_chatbot, t_input])
            t_clear.click(lambda: (clear_memory("text") or [], ""), None, [t_chatbot, t_input])

if __name__ == "__main__":
    demo.launch(server_port=7860, inbrowser=True)
