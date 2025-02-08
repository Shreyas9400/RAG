import os
import streamlit as st
import google.generativeai as genai
from groq import Groq 
from openai import OpenAI
import requests
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaEmbeddings
from langchain_core.embeddings import Embeddings
import torch

class ModelProvider:
    def __init__(self, provider, model, api_key=None):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self._setup_client()

    def _setup_client(self):
        try:
            if self.provider == "ollama":
                self.base_url = "http://localhost:11434/api/generate"
            elif self.provider == "gemini":
                genai.configure(api_key=self.api_key)
                self.client = genai.GenerativeModel(self.model)
            elif self.provider == "groq":
                self.client = Groq(api_key=self.api_key)
            elif self.provider == "openai":
                self.client = OpenAI(api_key=self.api_key)
        except Exception as e:
            if self.provider == "ollama":
                st.warning("Ollama is not installed locally. Some features may not be available.")
            else:
                st.error(f"Error setting up {self.provider}: {str(e)}")

    def generate_response(self, system_prompt, temperature=0.3, stream=True):
        try:
            if self.provider == "ollama":
                if not hasattr(self, 'base_url'):
                    raise Exception("Ollama is not available")
                return self._ollama_generate(system_prompt, temperature, stream)
            elif self.provider == "gemini":
                return self._gemini_generate(system_prompt, temperature)
            elif self.provider == "groq":
                return self._groq_generate(system_prompt, temperature, stream)
            elif self.provider == "openai":
                return self._openai_generate(system_prompt, temperature, stream)
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return None

    def _ollama_generate(self, system_prompt, temperature, stream):
        response = requests.post(
            self.base_url,
            json={
                "model": self.model,
                "prompt": system_prompt,
                "stream": stream,
                "options": {
                    "temperature": temperature,
                    "num_ctx": 4096
                }
            },
            stream=stream
        )
        return response

    def _gemini_generate(self, system_prompt, temperature):
        generation_config = {
            "temperature": temperature,
        }
        response = self.client.generate_content(system_prompt, generation_config=generation_config)
        return response.text

    def _groq_generate(self, system_prompt, temperature, stream):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": system_prompt}],
            temperature=temperature,
            stream=stream
        )
        return response

    def _openai_generate(self, system_prompt, temperature, stream):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": system_prompt}],
            temperature=temperature,
            stream=stream
        )
        return response