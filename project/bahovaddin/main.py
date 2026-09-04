import datetime
import os
import re
import time
import zoneinfo

from duckduckgo_search import DDGS
from google import genai
from google.genai import types
import gradio as gr
import pydub
import speech_recognition as sr



class Config:
    GEMINI_API_KEY = None
    TIMEZONE = "Asia/Dushanbe"

    @classmethod
    def load_key(cls):
        try:
            from google.colab import userdata
            cls.GEMINI_API_KEY = userdata.get("GEMINI_API_KEY")
            if cls.GEMINI_API_KEY:
                os.environ["GEMINI_API_KEY"] = cls.GEMINI_API_KEY
        except Exception:
            pass

        if not cls.GEMINI_API_KEY:
            cls.GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")


Config.load_key()
config = Config()

ai_client = None
if config.GEMINI_API_KEY:
    try:
        ai_client = genai.Client(api_key=config.GEMINI_API_KEY)
        print("✅ Gemini API успешно подключен!")
    except Exception as e:
        print(f"❌ Ошибка подключения Gemini: {e}")
else:
    print("⚠️ GEMINI_API_KEY не найден!")


def transcribe_audio_file(audio_path):
    if not audio_path:
        return ""
    recognizer = sr.Recognizer()
    try:
        sound = pydub.AudioSegment.from_file(audio_path)
        wav_path = "temp_gradio_audio.wav"
        sound.export(wav_path, format="wav")

        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data, language="ru-RU")

        if os.path.exists(wav_path):
            os.remove(wav_path)
        return text
    except Exception as e:
        print(f"❌ Ошибка распознавания речи: {e}")
        return ""


def search_web(query):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if results:
                formatted = [f"🔹 {r['title']}\n{r['body']}" for r in results]
                return "\n\n".join(formatted)
            return "Не удалось найти информацию по вашему запросу."
    except Exception as e:
        return f"Ошибка поиска: {str(e)}"


def ask_ai(prompt, history=None, files=None):
    if not ai_client:
        return "⚠️ API-ключ Gemini не найден. Проверьте секреты в Colab."

    system_instruction = (
        "Ты — Джарвис, вежливый, умный и преданный ИИ-помощник. "
        "Отвечай свободно на любые вопросы кратко и по делу на русском языке. "
        "Учитывай всю предыдущую историю беседы. "
        "Всегда обращайся к пользователю «Сэр». "
        "Поддерживай стиль высокотехнологичного ассистента. "
        "Выделяй ключевые факты и команды жирным шрифтом. "
        "Избегай вводной «воды» и сразу переходи к сути ответа."
    )

    full_prompt = system_instruction + "\n\n"
    if history:
        for msg in history:
            role = "Пользователь" if msg["role"] == "user" else "Джарвис"
            raw_text = msg.get("content", "")
            if isinstance(raw_text, list):
                text_str = "".join(
                    [str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in raw_text])
            else:
                text_str = str(raw_text)
            text_str = re.sub(r"^(📷 \[Фото прикреплено\]\n|🎙 )", "", text_str)
            if text_str.strip():
                full_prompt += f"{role}: {text_str}\n"

    full_prompt += f"Пользователь: {prompt}\nДжарвис:"

    candidate_models = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash-exp"]

    try:
        available_models = [m.name.replace("models/", "") for m in ai_client.models.list()]
        flash_models = [m for m in available_models if "flash" in m or "pro" in m]
        if flash_models:
            candidate_models = flash_models + candidate_models
    except Exception:
        pass
    last_error = ""
    for model_name in candidate_models:
        try:
            response = ai_client.models.generate_content(
                model=model_name,
                contents=full_prompt,
            )
            if response and hasattr(response, 'text') and response.text:
                return response.text.strip()
        except Exception as e:
            last_error = str(e)
            continue

    return f"Сэр, произошла ошибка API Gemini: {last_error}"


def execute_system_command(command, history=None, files=None):
    if not command and not files:
        return "Слушаю вас, Сэр."

    cmd_clean = command.strip().lower() if command else ""
    try:
        tz = zoneinfo.ZoneInfo(config.TIMEZONE)
        now = datetime.datetime.now(tz)
    except Exception:
        now = datetime.datetime.now()

    if "время" in cmd_clean or "который час" in cmd_clean:
        return f"Текущее время: {now.strftime('%H:%M:%S')}, Сэр."
    elif "дата" in cmd_clean or "какое число" in cmd_clean:
        return f"Сегодня {now.strftime('%d.%m.%Y')}."
    elif cmd_clean.startswith("найди в инете") or cmd_clean.startswith("гугл"):
        query = re.sub(r"^(найди в инете|гугл)\s*", "", command, flags=re.IGNORECASE).strip()
        return search_web(query)
    else:
        return ask_ai(command, history=history, files=files)


custom_css = """
body, .gradio-container {
    background-color: #02050c !important;
    max-width: 98% !important;
    width: 1400px !important;
    margin: 0 auto !important;
    font-family: 'Segoe UI', system-ui, sans-serif !important;
}

.chatbot {
    background: rgba(6, 11, 20, 0.85) !important;
    border: 1px solid rgba(0, 195, 255, 0.3) !important;
    border-radius: 14px !important;
    backdrop-filter: blur(10px) !important;
    box-shadow: 0 0 40px rgba(0, 162, 255, 0.12) !important;
    width: 100% !important;
}

/* Увеличен шрифт ответов в чате до 19px */
.chatbot .message, .chatbot p, .chatbot code, .chatbot span {
    font-size: 19px !important;
    line-height: 1.4 !important;
}

.chatbot button, 
.chatbot .options, 
.chatbot .message-buttons, 
.chatbot [class*="aria-label"],
.chatbot .action-buttons,
.chatbot .icons {
    display: none !important;
}

.jarvis-input-panel {
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
    background: rgba(10, 18, 30, 0.9) !important;
    border: 1px solid rgba(0, 195, 255, 0.35) !important;
    border-radius: 25px !important;
    padding: 6px 16px !important;
    margin-top: 8px !important;
    width: 100% !important;
}

.multimodal-textbox {
    flex-grow: 1 !important;
    border: none !important;
    background: transparent !important;
}

/* Увеличен шрифт поля ввода до 19px */
.multimodal-textbox textarea {
    color: #e2e8f0 !important;
    font-size: 19px !important;
    font-weight: 400 !important;
}

.mic-inline {
    max-width: 48px !important;
    min-width: 48px !important;
    height: 48px !important;
    background: #0088ff !important;
    border-radius: 50% !important;
    border: 1px solid rgba(255, 255, 255, 0.5) !important;
    box-shadow: 0 0 16px rgba(0, 136, 255, 0.8) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    padding: 0 !important;
    overflow: hidden !important;
}

.mic-inline label, .mic-inline span, .mic-inline div {
    display: none !important;
}

.mic-inline button {
    background: transparent !important;
    border: none !important;
    color: #ffffff !important;
    font-size: 20px !important;
    cursor: pointer !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 100% !important;
    height: 100% !important;
}

footer { display: none !important; }
"""
space_tech_canvas = """
<iframe srcdoc="
<!DOCTYPE html>
<html>
<head>
<style>
    body { margin: 0; padding: 0; background: transparent; overflow: hidden; display: flex; justify-content: center; align-items: center; }
    canvas { display: block; }
</style>
</head>
<body>
<canvas id='techCanvas' width='1200' height='320'></canvas>
<script>
    const canvas = document.getElementById('techCanvas');
    const ctx = canvas.getContext('2d');

    const width = canvas.width;
    const height = canvas.height;
    const centerX = width / 2;
    const centerY = height / 2;

    const stars = [];
    for (let i = 0; i < 220; i++) {
        stars.push({
            x: Math.random() * width,
            y: Math.random() * height,
            size: Math.random() * 1.6,
            alpha: Math.random(),
            speed: Math.random() * 0.02 + 0.005
        });
    }

    const techNodes = [];
    for (let i = 0; i < 40; i++) {
        techNodes.push({
            x: Math.random() * width,
            y: Math.random() * height,
            vx: (Math.random() - 0.5) * 0.5,
            vy: (Math.random() - 0.5) * 0.5
        });
    }

    const planetRadius = 75;
    const sphereParticles = [];
    for (let i = 0; i < 800; i++) {
        const u = Math.random();
        const v = Math.random();
        const theta = u * 2.0 * Math.PI;
        const phi = Math.acos(2.0 * v - 1.0);
        const r = planetRadius * (0.85 + Math.random() * 0.3);

        sphereParticles.push({
            x: r * Math.sin(phi) * Math.cos(theta),
            y: r * Math.sin(phi) * Math.sin(theta),
            z: r * Math.cos(phi),
            size: Math.random() * 1.4 + 0.5
        });
    }

    function createOrbitParticles(count, minR, maxR) {
        const arr = [];
        for (let i = 0; i < count; i++) {
            arr.push({
                a: Math.random() * Math.PI * 2,
                dist: minR + Math.random() * (maxR - minR),
                size: Math.random() * 1.6 + 0.4
            });
        }
        return arr;
    }

    const ringGray = createOrbitParticles(180, 110, 210);
    const ringYellow = createOrbitParticles(150, 100, 170);
    const ringOrange = createOrbitParticles(130, 90, 150);

    let angle = 0;

    function drawRotatedRing(ring, rot, color, tiltAngle, rotZAngle) {
        ctx.save();
        ctx.translate(centerX, centerY);
        ctx.rotate(rotZAngle);

        ctx.fillStyle = color;
        ring.forEach(p => {
            const currAngle = p.a + rot;
            const rx = Math.cos(currAngle) * p.dist;
            const rz = Math.sin(currAngle) * p.dist;
            const ry = rz * tiltAngle;

            const alpha = (rz + 200) / 400;
            ctx.globalAlpha = Math.max(0.15, Math.min(0.9, alpha));
            ctx.beginPath();
            ctx.arc(rx, ry, p.size, 0, Math.PI * 2);
            ctx.fill();
        });
        ctx.restore();
        ctx.globalAlpha = 1.0;
    }

    function draw() {
        ctx.clearRect(0, 0, width, height);
        angle += 0.012;

        stars.forEach(s => {
            s.alpha += s.speed;
            if (s.alpha > 1  s.alpha < 0) s.speed = -s.speed;
            ctx.fillStyle = `rgba(255, 255, 255, ${Math.abs(s.alpha)})`;
            ctx.beginPath();
            ctx.arc(s.x, s.y, s.size, 0, Math.PI * 2);
            ctx.fill();
        });

        for (let i = 0; i < techNodes.length; i++) {
            techNodes[i].x += techNodes[i].vx;
            techNodes[i].y += techNodes[i].vy;

            if (techNodes[i].x < 0  techNodes[i].x > width) techNodes[i].vx *= -1;
            if (techNodes[i].y < 0 || techNodes[i].y > height) techNodes[i].vy *= -1;

            ctx.fillStyle = 'rgba(0, 195, 255, 0.4)';
            ctx.beginPath();
            ctx.arc(techNodes[i].x, techNodes[i].y, 1.2, 0, Math.PI * 2);
            ctx.fill();
for (let j = i + 1; j < techNodes.length; j++) {
                const dx = techNodes[i].x - techNodes[j].x;
                const dy = techNodes[i].y - techNodes[j].y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < 130) {
                    ctx.strokeStyle = rgba(0, 195, 255, ${0.28 - dist / 500});
                    ctx.lineWidth = 0.8;
                    ctx.beginPath();
                    ctx.moveTo(techNodes[i].x, techNodes[i].y);
                    ctx.lineTo(techNodes[j].x, techNodes[j].y);
                    ctx.stroke();
                }
            }
        }

        drawRotatedRing(ringGray, angle, '#a0aec0', -0.25, -0.1);
        drawRotatedRing(ringYellow, -angle * 0.8, '#ecc94b', 0.45, Math.PI / 4);
        drawRotatedRing(ringOrange, angle * 1.1, '#ed8936', -0.5, -Math.PI / 3);

        sphereParticles.forEach(p => {
            const rotX = p.x * Math.cos(angle) - p.z * Math.sin(angle);
            const rotZ = p.x * Math.sin(angle) + p.z * Math.cos(angle);

            const scale = (rotZ + planetRadius * 2) / (planetRadius * 3);
            const px = centerX + rotX;
            const py = centerY + p.y;

            ctx.fillStyle = rgba(0, 212, 255, ${Math.max(0.2, scale)});
            ctx.beginPath();
            ctx.arc(px, py, p.size * scale, 0, Math.PI * 2);
            ctx.fill();
        });

        requestAnimationFrame(draw);
    }
    draw();
</script>
</body>
</html>
" style="width: 100%; height: 320px; border: none; background: transparent;"></iframe>
"""



def build_ui():
    with gr.Blocks(title="Jarvis Control Center") as demo:
        gr.HTML(space_tech_canvas)

        chatbot = gr.Chatbot(
            value=[],
            height=260,
            show_label=False,
            elem_classes=["chatbot"]
        )

        with gr.Row(elem_classes=["jarvis-input-panel"]):
            chat_input = gr.MultimodalTextbox(
                interactive=True,
                file_types=["image"],
                placeholder="Спросить Jarvis...",
                show_label=False,
                elem_classes=["multimodal-textbox"]
            )
            audio_input = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="",
                container=False,
                elem_classes=["mic-inline"]
            )

        def handle_user_input(user_data, history):
            text = user_data.get("text", "")
            files = user_data.get("files", [])

            if not text and not files:
                return None, history

            display_content = text
            if files:
                display_content = f"📷 [Фото прикреплено]\n{text}" if text else "📷 [Фото прикреплено]"

            reply = execute_system_command(text, history=history, files=files)

            history.append({"role": "user", "content": display_content})
            history.append({"role": "assistant", "content": reply})

            return None, history

        def handle_audio(audio_path, history):
            if not audio_path:
                return history
            rec_text = transcribe_audio_file(audio_path)
            if rec_text:
                reply = execute_system_command(rec_text, history=history)
                history.append({"role": "user", "content": f"🎙 {rec_text}"})
                history.append({"role": "assistant", "content": reply})
            return history

        chat_input.submit(handle_user_input, [chat_input, chatbot], [chat_input, chatbot])
        audio_input.stop_recording(handle_audio, [audio_input, chatbot], [chatbot])

    return demo


if name == "main":
    demo_app = build_ui()
    demo_app.launch(css=custom_css, theme=gr.themes.Soft(), share=True, debug=True)
                