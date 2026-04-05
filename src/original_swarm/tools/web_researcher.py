"""
Web Researcher tool for Swarm.
Integrates Crawl4AI and LLM-based autonomous navigation to find solutions when the agent is stuck.
"""

import asyncio
import logging
import os
import re
from typing import List, Dict, Any, Optional
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

class WebResearcher:
    """
    Herramienta de investigación web autónoma.
    
    Proposito: Extraer patrones de solución de fuentes de verdad (GitHub, StackOverflow, etc.)
    cuando los agentes internos no logran resolver un problema de programación.
    """

    def __init__(self, use_crawl4ai: bool = True):
        self.use_crawl4ai = use_crawl4ai
        self._user_agent = "Swarm-Autonomous-Agent/1.0"

    def search_and_extract(self, query: str, sites: List[str] = None) -> str:
        """
        Realiza una búsqueda y extrae el contenido más relevante.
        """
        if not sites:
            sites = ["github.com", "stackoverflow.com", "reddit.com/r/python"]
        
        search_query = f"{query} site:{' OR site:'.join(sites)}"
        logger.info(f"Investigando en la web: {search_query}")
        
        # Simulación de búsqueda (en producción usaría DuckDuckGo o Google Search API)
        # Por ahora extraemos directamente si nos pasan una URL o buscamos el patrón.
        return self._fallback_search(query)

    def _fallback_search(self, query: str) -> str:
        """Búsqueda simple usando DuckDuckGo Lite si no hay APIs configuradas."""
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=3))
                if not results:
                    return "No se encontraron resultados relevantes en la web."
                
                output = ["### Resultados de Investigación Web ###"]
                for r in results:
                    content = self.scrape_url(r['href'])
                    output.append(f"Fuente: {r['href']}\nContenido extratado:\n{content[:1000]}...")
                
                return "\n\n".join(output)
        except Exception as e:
            return f"Error en búsqueda web: {str(e)}"

    def scrape_url(self, url: str) -> str:
        """Extrae contenido limpio de una URL."""
        if self.use_crawl4ai:
            try:
                # Intento de uso de Crawl4AI si está instalado
                from crawl4ai import AsyncWebCrawler
                
                async def _crawl():
                    async with AsyncWebCrawler() as crawler:
                        result = await crawler.arun(url=url)
                        return result.markdown if result.markdown else result.extracted_content
                
                return asyncio.run(_crawl())
            except ImportError:
                logger.warning("Crawl4AI no detectado, usando extractor BeautifulSoup.")
        
        return self._basic_scrape(url)

    def _basic_scrape(self, url: str) -> str:
        """Extractor básico compatible con entornos sin dependencias pesadas."""
        try:
            resp = requests.get(url, headers={"User-Agent": self._user_agent}, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')
            
            # Limpiar elementos ruidosos
            for s in soup(['script', 'style', 'nav', 'footer', 'header']):
                s.decompose()
                
            text = soup.get_text(separator='\n')
            # Limpiar espacios en blanco excesivos
            clean_text = re.sub(r'\n\s*\n', '\n\n', text)
            return clean_text.strip()[:3000]
        except Exception as e:
            return f"Fallo al escrapear {url}: {str(e)}"

def research_tool(query: str) -> str:
    """Función wrapper para ser registrada como Tool en el Swarm."""
    researcher = WebResearcher()
    return researcher.search_and_extract(query)
