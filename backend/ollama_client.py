import json
import logging
import requests
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2:latest"):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = 120  # 2 minutes timeout for complex SQL generation
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Generate response from Ollama model.
        
        Args:
            prompt: The user prompt/message
            system_prompt: Optional system prompt for context
            **kwargs: Additional parameters (temperature, top_p, etc.)
            
        Returns:
            Dictionary containing the response and metadata
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temperature for deterministic SQL
                "top_p": 0.9,
                "top_k": 40,
            }
        }
        
        # Add system prompt if provided
        if system_prompt:
            payload["system"] = system_prompt
        
        # Override default options with any provided kwargs
        if "temperature" in kwargs:
            payload["options"]["temperature"] = kwargs["temperature"]
        if "top_p" in kwargs:
            payload["options"]["top_p"] = kwargs["top_p"]
        if "top_k" in kwargs:
            payload["options"]["top_k"] = kwargs["top_k"]
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API request failed: {e}")
            raise Exception(f"Ollama API error: {e}")
    
    def chat(self, messages: list, **kwargs) -> Dict[str, Any]:
        """
        Chat completion interface similar to OpenAI/Gemini.
        
        Args:
            messages: List of message dictionaries with role/content
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing the response
        """
        # Convert messages to prompt format expected by Ollama
        prompt_parts = []
        for msg in messages:
            if msg.get("role") == "system":
                continue  # Handled separately in generate()
            prompt_parts.append(f"{msg['role']}: {msg['content']}")
        
        prompt = "\n".join(prompt_parts)
        
        # Extract system prompt if present
        system_prompt = None
        for msg in messages:
            if msg.get("role") == "system":
                system_prompt = msg["content"]
                break
        
        return self.generate(prompt, system_prompt, **kwargs)

# Global instance for easy access
ollama_client = OllamaClient()