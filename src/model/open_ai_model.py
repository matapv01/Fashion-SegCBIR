from openai import OpenAI

class open_ai_model:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", base_url: str = "https://api.openai.com/v1"):
        """
        Khởi tạo client kết nối đến OpenAI API.
        :param api_key: Mã khóa API (sk-xxxxxx)
        :param base_url: URL base của API (có thể thay đổi nếu dùng proxy hoặc Azure)
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    # 💬 CHAT COMPLETION
    def chat(self, messages: list) -> str:
        """
        Gửi hội thoại đến mô hình Chat GPT.
        :param messages: Danh sách tin nhắn [{role, content}]
        :return: Nội dung trả lời của mô hình
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        return response.choices[0].message.content

    # 🖼️ IMAGE GENERATION
    def generate_image(self, prompt: str, model: str = "gpt-image-1") -> str:
        """
        Sinh ảnh từ mô tả văn bản.
        :param prompt: Mô tả ảnh
        :param model: Model tạo ảnh (gpt-image-1)
        :return: URL ảnh kết quả
        """
        image = self.client.images.generate(
            model=model,
            prompt=prompt
        )
        return image.data[0].url

    # 🔊 TEXT TO SPEECH
    def text_to_speech(self, text: str, output_path: str = "output.mp3",
                       model: str = "gpt-4o-mini-tts", voice: str = "alloy"):
        """
        Chuyển văn bản thành giọng nói.
        :param text: Văn bản cần chuyển
        :param output_path: Đường dẫn file âm thanh
        :param model: Model TTS
        :param voice: Giọng nói ('alloy', 'verse', 'nova'…)
        """
        with self.client.audio.speech.with_streaming_response.create(
            model=model,
            voice=voice,
            input=text
        ) as response:
            response.stream_to_file(output_path)
        print(f"✅ File âm thanh đã lưu tại: {output_path}")

    # 🎙️ SPEECH TO TEXT
    def speech_to_text(self, file_path: str,
                       model: str = "gpt-4o-mini-transcribe") -> str:
        """
        Nhận dạng giọng nói từ file âm thanh.
        :param file_path: Đường dẫn file âm thanh
        :param model: Model STT
        :return: Văn bản nhận dạng được
        """
        with open(file_path, "rb") as audio_file:
            transcription = self.client.audio.transcriptions.create(
                model=model,
                file=audio_file
            )
        return transcription.text
