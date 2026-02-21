"""
Orcest.ai - The Self-Adaptive LLM Orchestrator
Platform for reliable AI agents (Fork of LangChain)
"""

import os
import time
import base64
import json
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(
    title="Orcest.ai",
    description="The Self-Adaptive LLM Orchestrator platform for reliable AI agents",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for animation frames
app.mount("/static", StaticFiles(directory="app/static"), name="static")

RAINYMODEL_BASE_URL = os.getenv("RAINYMODEL_BASE_URL", "https://rm.orcest.ai/v1")
SSO_ISSUER = os.getenv("SSO_ISSUER", "https://login.orcest.ai")
SSO_CLIENT_ID = os.getenv("SSO_CLIENT_ID", "orcest")
SSO_CLIENT_SECRET = os.getenv("SSO_CLIENT_SECRET")
SSO_CALLBACK_URL = os.getenv("SSO_CALLBACK_URL", "https://orcest.ai/auth/callback")
ORCEST_SSO_COOKIE = "orcest_sso_token"

LANGCHAIN_ECOSYSTEM = {
    "deep_agents": {
        "name": "Deep Agents",
        "description": "Build agents that can plan, use subagents, and leverage file systems for complex tasks.",
        "url": "https://docs.langchain.com/oss/python/deepagents/overview",
    },
    "langgraph": {
        "name": "LangGraph",
        "description": "Low-level orchestration framework for reliable, stateful, human-in-the-loop agents.",
        "url": "https://www.langchain.com/langgraph",
    },
    "integrations": {
        "name": "Integrations",
        "description": "Catalog of chat models, embedding models, tools, toolkits, and connectors.",
        "url": "https://docs.langchain.com/oss/python/integrations/providers/overview",
    },
    "langsmith": {
        "name": "LangSmith",
        "description": "Observability, evals, debugging, and production visibility for LLM applications.",
        "url": "https://www.langchain.com/langsmith",
    },
    "langsmith_deployment": {
        "name": "LangSmith Deployment",
        "description": "Purpose-built deployment for long-running, stateful workflows with fast iteration.",
        "url": "https://docs.langchain.com/langsmith/deployments",
    },
}

LANGCHAIN_ORCEST_ENDPOINTS = {
    "ecosystem": "https://orcest.ai/api/langchain/ecosystem",
    "health": "https://orcest.ai/api/langchain/health",
    "run_agent": "https://orcest.ai/api/langchain/agent/run",
    "rainymodel_manifest": "https://orcest.ai/api/rainymodel/langchain-manifest",
    "deep_agents": "https://orcest.ai/api/langchain/deep-agents",
    "langgraph": "https://orcest.ai/api/langchain/langgraph",
    "integrations": "https://orcest.ai/api/langchain/integrations",
    "langsmith": "https://orcest.ai/api/langchain/langsmith",
    "langsmith_deployment": "https://orcest.ai/api/langchain/langsmith-deployment",
    "langchain_ui": "https://orcest.ai/orchestration/langchain",
    "rainymodel_ui": "https://orcest.ai/orchestration/rainymodel",
    "rainymodel_manifest_schema": "https://orcest.ai/api/rainymodel/langchain-manifest/schema",
}


class LangChainAgentRunRequest(BaseModel):
    query: str
    agent_type: str = "general"
    metadata: dict | None = None

LANDING_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Orcest AI - The Self-Adaptive LLM Orchestrator</title>
<meta name="description" content="Intelligent LLM Orchestration Platform - unified routing across free, internal, and premium AI models. Complete AI ecosystem for developers.">
<meta name="keywords" content="AI, LLM, orchestration, chatbot, IDE, agent, artificial intelligence">
<script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-2701288361875881" crossorigin="anonymous"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg-primary:#0a0a0f;
  --bg-secondary:#111827;
  --bg-tertiary:#1f2937;
  --text-primary:#f8fafc;
  --text-secondary:#e2e8f0;
  --text-muted:#94a3b8;
  --accent-blue:#60a5fa;
  --accent-purple:#a78bfa;
  --accent-pink:#f472b6;
  --accent-green:#4ade80;
  --accent-yellow:#fbbf24;
  --accent-orange:#fb923c;
  --accent-cyan:#38bdf8;
  --border-color:#374151;
  --shadow-glow:0 0 50px rgba(96,165,250,0.15);
}
body{font-family:'Inter',system-ui,-apple-system,sans-serif;background:var(--bg-primary);color:var(--text-secondary);min-height:100vh;line-height:1.6}

/* Navigation */
.nav{position:fixed;top:0;left:0;right:0;z-index:100;background:rgba(10,10,15,0.95);backdrop-filter:blur(20px);border-bottom:1px solid var(--border-color)}
.nav-container{max-width:1200px;margin:0 auto;padding:0 20px;display:flex;justify-content:space-between;align-items:center;height:70px}
.nav-logo{font-size:1.5rem;font-weight:800;background:linear-gradient(135deg,var(--accent-blue),var(--accent-purple));-webkit-background-clip:text;-webkit-text-fill-color:transparent;text-decoration:none}
.nav-links{display:flex;gap:32px;align-items:center}
.nav-links a{color:var(--text-muted);text-decoration:none;font-weight:500;transition:color 0.3s}
.nav-links a:hover{color:var(--accent-blue)}
.nav-cta{background:linear-gradient(135deg,var(--accent-blue),var(--accent-purple));color:white;padding:10px 20px;border-radius:25px;text-decoration:none;font-weight:600;transition:transform 0.3s,box-shadow 0.3s}
.nav-cta:hover{transform:translateY(-2px);box-shadow:var(--shadow-glow)}

/* Hero Section */
.hero{position:relative;text-align:center;padding:140px 20px 80px;background:linear-gradient(135deg,#0a0a1a 0%,#1a1a3a 30%,#2a1a3a 70%,#0a0a1a 100%);overflow:hidden;min-height:100vh;display:flex;align-items:center;justify-content:center}
.hero-bg-animation{position:absolute;top:0;left:0;width:100%;height:100%;opacity:0.12;z-index:1;background-size:cover;background-position:center;animation:orcestAnimation 12s infinite linear}
@keyframes orcestAnimation{0%{background-image:url('/static/frames/frame-001.jpg')}12.5%{background-image:url('/static/frames/frame-026.jpg')}25%{background-image:url('/static/frames/frame-051.jpg')}37.5%{background-image:url('/static/frames/frame-076.jpg')}50%{background-image:url('/static/frames/key-frame-100.jpg')}62.5%{background-image:url('/static/frames/frame-126.jpg')}75%{background-image:url('/static/frames/frame-151.jpg')}87.5%{background-image:url('/static/frames/frame-176.jpg')}100%{background-image:url('/static/frames/key-frame-200.jpg')}}
.hero-content{position:relative;z-index:2;max-width:900px}
.hero h1{font-size:4.5rem;font-weight:800;background:linear-gradient(135deg,var(--accent-blue),var(--accent-purple),var(--accent-pink));-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:24px;line-height:1.1}
.hero-subtitle{font-size:1.4rem;color:var(--text-muted);max-width:700px;margin:0 auto 32px;font-weight:400}
.hero-description{font-size:1.1rem;color:var(--text-secondary);max-width:600px;margin:0 auto 40px;opacity:0.9}
.hero-cta{display:flex;gap:16px;justify-content:center;flex-wrap:wrap;margin-bottom:40px}
.btn-primary{background:linear-gradient(135deg,var(--accent-blue),var(--accent-purple));color:white;padding:16px 32px;border-radius:50px;text-decoration:none;font-weight:600;font-size:1.1rem;transition:all 0.3s;display:inline-flex;align-items:center;gap:8px}
.btn-primary:hover{transform:translateY(-3px);box-shadow:var(--shadow-glow)}
.btn-secondary{background:rgba(255,255,255,0.1);color:var(--text-primary);padding:16px 32px;border-radius:50px;text-decoration:none;font-weight:600;font-size:1.1rem;transition:all 0.3s;backdrop-filter:blur(10px);border:1px solid rgba(255,255,255,0.1)}
.btn-secondary:hover{background:rgba(255,255,255,0.15);transform:translateY(-2px)}
.hero-stats{display:flex;gap:40px;justify-content:center;flex-wrap:wrap;margin-top:60px}
.stat{text-align:center}
.stat-number{font-size:2.5rem;font-weight:800;color:var(--accent-blue);display:block}
.stat-label{font-size:0.9rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:1px}

/* Ecosystem Section */
.ecosystem{padding:100px 20px;background:linear-gradient(180deg,var(--bg-primary) 0%,var(--bg-secondary) 100%)}
.ecosystem-container{max-width:1400px;margin:0 auto}
.section-header{text-align:center;margin-bottom:80px}
.section-title{font-size:3rem;font-weight:800;color:var(--text-primary);margin-bottom:16px}
.section-subtitle{font-size:1.2rem;color:var(--text-muted);max-width:600px;margin:0 auto}
.services-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(350px,1fr));gap:32px;margin-bottom:80px}
.service-card{background:linear-gradient(135deg,var(--bg-secondary),var(--bg-tertiary));border:1px solid var(--border-color);border-radius:24px;padding:40px;transition:all 0.4s;text-decoration:none;color:inherit;display:block;position:relative;overflow:hidden}
.service-card::before{content:'';position:absolute;top:0;left:0;right:0;height:4px;background:linear-gradient(90deg,var(--accent-blue),var(--accent-purple),var(--accent-pink));opacity:0;transition:opacity 0.3s}
.service-card:hover::before{opacity:1}
.service-card:hover{border-color:var(--accent-blue);transform:translateY(-8px);box-shadow:0 20px 60px rgba(96,165,250,0.15)}
.service-header{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:20px}
.service-tag{display:inline-block;font-size:0.75rem;padding:6px 14px;border-radius:20px;margin-bottom:16px;font-weight:700;text-transform:uppercase;letter-spacing:0.5px}
.service-status{width:12px;height:12px;border-radius:50%;background:var(--accent-green);box-shadow:0 0 10px var(--accent-green)}
.service-title{font-size:1.5rem;color:var(--text-primary);margin-bottom:12px;font-weight:700}
.service-description{color:var(--text-muted);font-size:1rem;line-height:1.6;margin-bottom:20px}
.service-url{color:var(--accent-blue);font-size:0.9rem;font-weight:600;display:flex;align-items:center;gap:8px}
.service-features{display:flex;flex-wrap:wrap;gap:8px;margin-top:16px}
.feature-tag{background:rgba(96,165,250,0.1);color:var(--accent-blue);padding:4px 10px;border-radius:12px;font-size:0.8rem;font-weight:500}

/* Tag Colors */
.tag-llm{background:rgba(56,189,248,0.15);color:var(--accent-cyan)}
.tag-chat{background:rgba(74,222,128,0.15);color:var(--accent-green)}
.tag-agent{background:rgba(251,191,36,0.15);color:var(--accent-yellow)}
.tag-ide{background:rgba(196,132,252,0.15);color:var(--accent-purple)}
.tag-api{background:rgba(96,165,250,0.15);color:var(--accent-blue)}
.tag-sso{background:rgba(251,146,60,0.15);color:var(--accent-orange)}
.tag-status{background:rgba(74,222,128,0.15);color:var(--accent-green)}

/* Features Section */
.features{padding:100px 20px;background:var(--bg-primary)}
.features-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:40px;max-width:1200px;margin:0 auto}
.feature{text-align:center;padding:40px 20px}
.feature-icon{width:80px;height:80px;margin:0 auto 24px;background:linear-gradient(135deg,var(--accent-blue),var(--accent-purple));border-radius:20px;display:flex;align-items:center;justify-content:center;font-size:2rem}
.feature h3{font-size:1.4rem;color:var(--text-primary);margin-bottom:16px;font-weight:700}
.feature p{color:var(--text-muted);line-height:1.6}

/* Footer */
.footer{background:var(--bg-secondary);border-top:1px solid var(--border-color);padding:60px 20px 40px}
.footer-container{max-width:1200px;margin:0 auto}
.footer-content{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:40px;margin-bottom:40px}
.footer-section h4{color:var(--text-primary);font-size:1.2rem;font-weight:700;margin-bottom:20px}
.footer-section a{color:var(--text-muted);text-decoration:none;display:block;margin-bottom:8px;transition:color 0.3s}
.footer-section a:hover{color:var(--accent-blue)}
.footer-bottom{text-align:center;padding-top:40px;border-top:1px solid var(--border-color);color:var(--text-muted)}
.footer-bottom a{color:var(--accent-blue);text-decoration:none}

/* Responsive */
@media (prefers-reduced-motion: reduce){.hero-bg-animation{animation:none;background-image:url('/static/frames/key-frame-100.jpg')}}
@media (max-width: 1024px){.hero h1{font-size:3.5rem}.services-grid{grid-template-columns:repeat(auto-fit,minmax(300px,1fr))}}
@media (max-width: 768px){.nav-links{display:none}.hero{padding:120px 20px 60px}.hero h1{font-size:2.8rem}.hero-bg-animation{opacity:0.06}.hero-cta{flex-direction:column;align-items:center}.hero-stats{gap:20px}.services-grid{grid-template-columns:1fr;gap:24px}.service-card{padding:30px}}
@media (max-width: 480px){.hero h1{font-size:2.2rem}.btn-primary,.btn-secondary{padding:14px 24px;font-size:1rem}}
</style>
<script>
document.addEventListener('DOMContentLoaded', function() {
    // Preload animation frames
    const keyFrames = [
        '/static/frames/key-frame-001.jpg',
        '/static/frames/key-frame-050.jpg', 
        '/static/frames/key-frame-100.jpg',
        '/static/frames/key-frame-150.jpg',
        '/static/frames/key-frame-200.jpg'
    ];
    
    keyFrames.forEach(src => {
        const img = new Image();
        img.src = src;
    });
    
    // Handle reduced motion
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        const animationEl = document.querySelector('.hero-bg-animation');
        if (animationEl) {
            animationEl.style.animationPlayState = 'paused';
        }
    }
    
    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth' });
            }
        });
    });
    
    // Add loading states for external links
    document.querySelectorAll('a[href^="http"]').forEach(link => {
        link.addEventListener('click', function(e) {
            if (!this.target) {
                this.style.opacity = '0.7';
                this.innerHTML += ' <span style="font-size:0.8em">↗</span>';
            }
        });
    });
});
</script>
</head>
<body>
<!-- Navigation -->
<nav class="nav">
<div class="nav-container">
<a href="#" class="nav-logo">Orcest AI</a>
<div class="nav-links">
<a href="#ecosystem">Ecosystem</a>
<a href="#features">Features</a>
<a href="https://status.orcest.ai">Status</a>
<a href="https://github.com/langchain-ai/langchain" target="_blank" rel="noopener">LangChain</a>
<a href="https://login.orcest.ai" class="nav-cta">Login</a>
</div>
</div>
</nav>

<!-- Hero Section -->
<section class="hero">
<div class="hero-bg-animation"></div>
<div class="hero-content">
<h1>Orcest AI</h1>
<p class="hero-subtitle">The Self-Adaptive LLM Orchestrator</p>
<p class="hero-description">Intelligent orchestration platform unifying free, internal, and premium AI models with advanced routing, real-time processing, and comprehensive developer tools.</p>
<div class="hero-cta">
<a href="https://rm.orcest.ai" class="btn-primary">Start Building →</a>
<a href="#ecosystem" class="btn-secondary">Explore Ecosystem</a>
</div>
<div class="hero-stats">
<div class="stat">
<span class="stat-number">7+</span>
<span class="stat-label">AI Services</span>
</div>
<div class="stat">
<span class="stat-number">24/7</span>
<span class="stat-label">Uptime</span>
</div>
<div class="stat">
<span class="stat-number">∞</span>
<span class="stat-label">Possibilities</span>
</div>
</div>
</div>
</section>

<!-- Ecosystem Section -->
<section class="ecosystem" id="ecosystem">
<div class="ecosystem-container">
<div class="section-header">
<h2 class="section-title">Complete AI Ecosystem</h2>
<p class="section-subtitle">Integrated suite of AI-powered tools and services for modern development workflows</p>
</div>

<div class="services-grid">
<!-- RainyModel -->
<a href="https://rm.orcest.ai" class="service-card">
<div class="service-header">
<span class="service-tag tag-llm">LLM Proxy</span>
<div class="service-status"></div>
</div>
<h3 class="service-title">RainyModel</h3>
<p class="service-description">Intelligent LLM routing proxy that automatically distributes requests across free, internal, and premium AI backends with load balancing and failover.</p>
<span class="service-url">rm.orcest.ai →</span>
<div class="service-features">
<span class="feature-tag">OpenAI Compatible</span>
<span class="feature-tag">Multi-Provider</span>
<span class="feature-tag">Auto-Routing</span>
</div>
</a>

<!-- Lamino -->
<a href="https://llm.orcest.ai" class="service-card">
<div class="service-header">
<span class="service-tag tag-chat">AI Chat</span>
<div class="service-status"></div>
</div>
<h3 class="service-title">Lamino</h3>
<p class="service-description">Advanced AI chat interface with RAG capabilities, document processing, workspace management, and collaborative features powered by RainyModel.</p>
<span class="service-url">llm.orcest.ai →</span>
<div class="service-features">
<span class="feature-tag">RAG Support</span>
<span class="feature-tag">Document AI</span>
<span class="feature-tag">Workspaces</span>
</div>
</a>

<!-- Maestrist -->
<a href="https://agent.orcest.ai" class="service-card">
<div class="service-header">
<span class="service-tag tag-agent">AI Agent</span>
<div class="service-status"></div>
</div>
<h3 class="service-title">Maestrist</h3>
<p class="service-description">Autonomous software development agent capable of coding, debugging, testing, and project management with advanced reasoning capabilities.</p>
<span class="service-url">agent.orcest.ai →</span>
<div class="service-features">
<span class="feature-tag">Auto-Coding</span>
<span class="feature-tag">Debug AI</span>
<span class="feature-tag">Project Mgmt</span>
</div>
</a>

<!-- Orcide -->
<a href="https://ide.orcest.ai" class="service-card">
<div class="service-header">
<span class="service-tag tag-ide">AI IDE</span>
<div class="service-status"></div>
</div>
<h3 class="service-title">Orcide</h3>
<p class="service-description">AI-powered integrated development environment with intelligent autocomplete, inline chat assistance, and advanced code generation capabilities.</p>
<span class="service-url">ide.orcest.ai →</span>
<div class="service-features">
<span class="feature-tag">Smart Autocomplete</span>
<span class="feature-tag">Inline Chat</span>
<span class="feature-tag">Code Gen</span>
</div>
</a>

<!-- Core Orchestration -->
<a href="https://login.orcest.ai/oauth2/authorize?client_id=orcest&redirect_uri=https%3A%2F%2Forcest.ai%2Fauth%2Fcallback&response_type=code&scope=openid%20profile%20email&state=eyJyZXR1cm5UbyI6Ii9vcmNoZXN0cmF0aW9uIn0" class="service-card">
<div class="service-header">
<span class="service-tag tag-api">Core API</span>
<div class="service-status"></div>
</div>
<h3 class="service-title">Orcest Core</h3>
<p class="service-description">LangChain-based orchestration engine providing real-time search, data extraction, research automation, and web crawling with SSO authentication.</p>
<span class="service-url">orcest.ai/orchestration →</span>
<div class="service-features">
<span class="feature-tag">LangChain</span>
<span class="feature-tag">Real-time Search</span>
<span class="feature-tag">SSO Required</span>
</div>
</a>

<!-- System Status -->
<a href="https://status.orcest.ai" class="service-card">
<div class="service-header">
<span class="service-tag tag-status">Monitoring</span>
<div class="service-status"></div>
</div>
<h3 class="service-title">System Status</h3>
<p class="service-description">Real-time monitoring dashboard with service health, quality audits, architecture diagrams, and comprehensive ecosystem status tracking.</p>
<span class="service-url">status.orcest.ai →</span>
<div class="service-features">
<span class="feature-tag">Live Monitoring</span>
<span class="feature-tag">Health Checks</span>
<span class="feature-tag">Analytics</span>
</div>
</a>

<!-- SSO Portal -->
<a href="https://login.orcest.ai" class="service-card">
<div class="service-header">
<span class="service-tag tag-sso">Authentication</span>
<div class="service-status"></div>
</div>
<h3 class="service-title">SSO Portal</h3>
<p class="service-description">Centralized single sign-on authentication system for all Orcest AI services with account management and API key administration.</p>
<span class="service-url">login.orcest.ai →</span>
<div class="service-features">
<span class="feature-tag">Single Sign-On</span>
<span class="feature-tag">API Keys</span>
<span class="feature-tag">Account Mgmt</span>
</div>
</a>

<!-- LangChain Ecosystem -->
<a href="/api/langchain/ecosystem" class="service-card">
<div class="service-header">
<span class="service-tag tag-api">LangChain</span>
<div class="service-status"></div>
</div>
<h3 class="service-title">LangChain Ecosystem</h3>
<p class="service-description">Deep Agents, LangGraph, Integrations, LangSmith, and Deployment endpoints are available directly via Orcest AI APIs for unified orchestration.</p>
<span class="service-url">orcest.ai/api/langchain/ecosystem →</span>
<div class="service-features">
<span class="feature-tag">Deep Agents</span>
<span class="feature-tag">LangGraph</span>
<span class="feature-tag">LangSmith</span>
</div>
</a>
</div>
</div>
</section>

<!-- Features Section -->
<section class="features" id="features">
<div class="section-header">
<h2 class="section-title">Why Choose Orcest AI?</h2>
<p class="section-subtitle">Built for developers, by developers - with enterprise-grade reliability</p>
</div>
<div class="features-grid">
<div class="feature">
<div class="feature-icon">🚀</div>
<h3>Lightning Fast</h3>
<p>Optimized routing and caching ensure sub-second response times across all services with intelligent load balancing.</p>
</div>
<div class="feature">
<div class="feature-icon">🔒</div>
<h3>Enterprise Security</h3>
<p>End-to-end encryption, SSO integration, and comprehensive audit trails meet enterprise security requirements.</p>
</div>
<div class="feature">
<div class="feature-icon">🔧</div>
<h3>Developer First</h3>
<p>RESTful APIs, comprehensive documentation, and SDKs in multiple languages for seamless integration.</p>
</div>
<div class="feature">
<div class="feature-icon">📊</div>
<h3>Real-time Analytics</h3>
<p>Detailed usage metrics, performance monitoring, and cost optimization insights across your AI operations.</p>
</div>
<div class="feature">
<div class="feature-icon">🌐</div>
<h3>Multi-Provider</h3>
<p>Unified interface to OpenAI, Anthropic, Google, and 20+ AI providers with automatic failover and cost optimization.</p>
</div>
<div class="feature">
<div class="feature-icon">⚡</div>
<h3>Auto-Scaling</h3>
<p>Dynamic scaling based on demand with intelligent resource allocation and cost-effective usage patterns.</p>
</div>
</div>
</section>

<!-- Footer -->
<footer class="footer">
<div class="footer-container">
<div class="footer-content">
<div class="footer-section">
<h4>Platform</h4>
<a href="https://rm.orcest.ai">RainyModel Proxy</a>
<a href="https://llm.orcest.ai">Lamino Chat</a>
<a href="https://agent.orcest.ai">Maestrist Agent</a>
<a href="https://ide.orcest.ai">Orcide IDE</a>
</div>
<div class="footer-section">
<h4>Resources</h4>
<a href="https://github.com/langchain-ai/langchain" target="_blank" rel="noopener">LangChain</a>
<a href="https://status.orcest.ai">System Status</a>
<a href="https://login.orcest.ai">Authentication</a>
<a href="/api/info">API Documentation</a>
<a href="/ecosystem/health">Health Check</a>
</div>
<div class="footer-section">
<h4>Company</h4>
<a href="mailto:admin@danial.ai">Contact Support</a>
<a href="https://github.com/danialsamiei">GitHub</a>
<a href="/metrics">System Metrics</a>
<a href="#ecosystem">Ecosystem</a>
</div>
<div class="footer-section">
<h4>Connect</h4>
<a href="https://twitter.com/orcest_ai">Twitter</a>
<a href="https://linkedin.com/company/orcest-ai">LinkedIn</a>
<a href="mailto:hello@orcest.ai">Email Us</a>
<a href="https://discord.gg/orcest">Discord</a>
</div>
</div>
<div class="footer-bottom">
<p>&copy; 2025 Orcest AI. The Self-Adaptive LLM Orchestrator. Built with ❤️ for developers.</p>
<p>Support: <a href="mailto:admin@danial.ai">admin@danial.ai</a> | Status: <a href="https://status.orcest.ai">All Systems Operational</a></p>
</div>
</div>
</footer>
</body>
</html>"""


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "orcest.ai"}


@app.get("/", response_class=HTMLResponse)
async def landing_page():
    return HTMLResponse(content=LANDING_HTML)


ECOSYSTEM_SERVICES = [
    {"name": "orcest.ai", "url": "https://orcest.ai/health"},
    {"name": "rm.orcest.ai", "url": "https://rm.orcest.ai/health"},
    {"name": "llm.orcest.ai", "url": "https://llm.orcest.ai/api/health"},
    {"name": "agent.orcest.ai", "url": "https://agent.orcest.ai/api/litellm-models"},
    {"name": "ide.orcest.ai", "url": "https://ide.orcest.ai"},
    {"name": "login.orcest.ai", "url": "https://login.orcest.ai/health"},
    {"name": "status.orcest.ai", "url": "https://status-orcest-ai.onrender.com/health"},
]

_metrics = {"requests": 0, "start_time": time.time()}


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    _metrics["requests"] += 1
    response = await call_next(request)
    return response


@app.get("/api/info")
async def api_info():
    return {
        "platform": "orcest.ai",
        "description": "Intelligent LLM Orchestration Platform",
        "services": {
            "rainymodel": "https://rm.orcest.ai",
            "lamino": "https://llm.orcest.ai",
            "maestrist": "https://agent.orcest.ai",
            "orcide": "https://ide.orcest.ai",
            "login": "https://login.orcest.ai",
            "status": "https://status.orcest.ai",
            "langchain_ecosystem": "https://orcest.ai/api/langchain/ecosystem",
        },
        "rainymodel_api": RAINYMODEL_BASE_URL,
        "langchain_api_endpoints": LANGCHAIN_ORCEST_ENDPOINTS,
    }


@app.get("/api/langchain/health")
async def langchain_health():
    return {
        "status": "healthy",
        "provider": "orcest.ai",
        "service": "langchain-api",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoints": LANGCHAIN_ORCEST_ENDPOINTS,
    }


@app.get("/api/langchain/ecosystem")
async def langchain_ecosystem():
    return {
        "name": "LangChain Ecosystem via Orcest.ai",
        "components": LANGCHAIN_ECOSYSTEM,
        "orcest_endpoints": LANGCHAIN_ORCEST_ENDPOINTS,
        "notes": {
            "integration_mode": "orcest.ai acts as unified discovery and orchestration surface",
            "rainymodel_connection": "RainyModel can consume Orcest LangChain endpoint manifest",
        },
    }


def _langchain_component_response(component_key: str):
    component = LANGCHAIN_ECOSYSTEM[component_key]
    return {
        "component": component,
        "access": {
            "type": "internal_endpoint",
            "endpoint": LANGCHAIN_ORCEST_ENDPOINTS[component_key],
            "method": "GET",
            "sso_ui": LANGCHAIN_ORCEST_ENDPOINTS["langchain_ui"],
        },
        "rainymodel": {
            "proxy_base": RAINYMODEL_BASE_URL,
            "recommended_upstream": LANGCHAIN_ORCEST_ENDPOINTS["run_agent"],
            "component_hint": component_key,
        },
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/langchain/deep-agents")
async def langchain_deep_agents():
    return _langchain_component_response("deep_agents")


@app.get("/api/langchain/langgraph")
async def langchain_langgraph():
    return _langchain_component_response("langgraph")


@app.get("/api/langchain/integrations")
async def langchain_integrations():
    return _langchain_component_response("integrations")


@app.get("/api/langchain/langsmith")
async def langchain_langsmith():
    return _langchain_component_response("langsmith")


@app.get("/api/langchain/langsmith-deployment")
async def langchain_langsmith_deployment():
    return _langchain_component_response("langsmith_deployment")


@app.post("/api/langchain/agent/run")
async def langchain_agent_run(payload: LangChainAgentRunRequest):
    return {
        "status": "accepted",
        "engine": "langchain-orcest",
        "agent_type": payload.agent_type,
        "query": payload.query,
        "metadata": payload.metadata or {},
        "next": {
            "rainymodel_proxy": f"{RAINYMODEL_BASE_URL}/chat/completions",
            "ecosystem": LANGCHAIN_ORCEST_ENDPOINTS["ecosystem"],
        },
    }


@app.get("/api/rainymodel/langchain-manifest")
async def rainymodel_langchain_manifest():
    return {
        "manifest_version": "1.1.0",
        "schema_endpoint": LANGCHAIN_ORCEST_ENDPOINTS["rainymodel_manifest_schema"],
        "consumer": "RainyModel",
        "target": "Orcest LangChain API",
        "recommended_endpoint": LANGCHAIN_ORCEST_ENDPOINTS["run_agent"],
        "health_endpoint": LANGCHAIN_ORCEST_ENDPOINTS["health"],
        "ecosystem_endpoint": LANGCHAIN_ORCEST_ENDPOINTS["ecosystem"],
        "routing_policy_hint": "RainyModel should include Orcest LangChain API as one upstream endpoint",
        "auth": {
            "required": True,
            "mode": "SSO for console, API key for service-to-service",
            "issuer": SSO_ISSUER,
        },
        "capabilities": {
            "supports_component_discovery": True,
            "supports_health_probe": True,
            "supports_agent_execution": True,
            "supports_ui_console": True,
        },
        "examples": {
            "agent_run_post": {
                "url": LANGCHAIN_ORCEST_ENDPOINTS["run_agent"],
                "method": "POST",
                "body": {"query": "Plan a multi-step migration", "agent_type": "general", "metadata": {"source": "RainyModel"}},
            },
            "health_get": {
                "url": LANGCHAIN_ORCEST_ENDPOINTS["health"],
                "method": "GET",
            },
        },
        "access_map": {
            "deep_agents": {
                "endpoint": LANGCHAIN_ORCEST_ENDPOINTS["deep_agents"],
                "method": "GET",
                "description": "Planning-oriented agent access with subagent and filesystem patterns.",
            },
            "langgraph": {
                "endpoint": LANGCHAIN_ORCEST_ENDPOINTS["langgraph"],
                "method": "GET",
                "description": "Reliable low-level graph orchestration and stateful workflow access.",
            },
            "integrations": {
                "endpoint": LANGCHAIN_ORCEST_ENDPOINTS["integrations"],
                "method": "GET",
                "description": "Provider/tool integration catalog for model and toolkit discovery.",
            },
            "langsmith": {
                "endpoint": LANGCHAIN_ORCEST_ENDPOINTS["langsmith"],
                "method": "GET",
                "description": "Observability, eval, and production debugging access surface.",
            },
            "langsmith_deployment": {
                "endpoint": LANGCHAIN_ORCEST_ENDPOINTS["langsmith_deployment"],
                "method": "GET",
                "description": "Deployment and scale guidance for long-running stateful agent workflows.",
            },
            "agent_run": {
                "endpoint": LANGCHAIN_ORCEST_ENDPOINTS["run_agent"],
                "method": "POST",
                "description": "Unified execution entrypoint that RainyModel can call as an upstream.",
            },
        },
        "ui_console": LANGCHAIN_ORCEST_ENDPOINTS["langchain_ui"],
        "rainymodel_console": LANGCHAIN_ORCEST_ENDPOINTS["rainymodel_ui"],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/rainymodel/langchain-manifest/schema")
async def rainymodel_langchain_manifest_schema():
    return {
        "name": "RainyModel LangChain Manifest Schema",
        "version": "1.1.0",
        "required_fields": [
            "manifest_version",
            "consumer",
            "target",
            "recommended_endpoint",
            "health_endpoint",
            "ecosystem_endpoint",
            "access_map",
            "updated_at",
        ],
        "access_map_item_schema": {
            "endpoint": "string (absolute URL)",
            "method": "GET|POST",
            "description": "string",
        },
        "notes": "RainyModel can validate manifest payload shape before consuming endpoint map.",
    }


@app.get("/ecosystem/health")
async def ecosystem_health():
    async with httpx.AsyncClient(follow_redirects=True, timeout=10.0) as client:
        results = {}
        for svc in ECOSYSTEM_SERVICES:
            try:
                resp = await client.get(svc["url"])
                results[svc["name"]] = {"status": "operational" if resp.status_code < 400 else "degraded", "code": resp.status_code}
            except Exception:
                results[svc["name"]] = {"status": "down", "code": 0}
    operational = sum(1 for v in results.values() if v["status"] == "operational")
    return {
        "overall": "operational" if operational == len(results) else "degraded",
        "services": results,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/metrics")
async def metrics_endpoint():
    uptime = time.time() - _metrics["start_time"]
    return {
        "uptime_seconds": int(uptime),
        "total_requests": _metrics["requests"],
        "service": "orcest.ai",
        "version": "1.0.0",
    }


# --- Orcest AI SSO Auth ---

@app.get("/auth/callback")
async def auth_callback(request: Request, response: Response, code: str = "", state: str = ""):
    """OAuth2 callback from login.orcest.ai - exchange code for token and set cookie"""
    if not SSO_CLIENT_SECRET or not code:
        return RedirectResponse(url="/", status_code=302)

    return_to = "/orchestration"
    if state:
        try:
            decoded = base64.urlsafe_b64decode(state + "==")
            data = json.loads(decoded)
            return_to = data.get("returnTo", "/orchestration")
        except Exception:
            pass

    async with httpx.AsyncClient() as client:
        token_res = await client.post(
            f"{SSO_ISSUER}/oauth2/token",
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": SSO_CALLBACK_URL,
                "client_id": SSO_CLIENT_ID,
                "client_secret": SSO_CLIENT_SECRET,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

    if token_res.status_code != 200:
        return RedirectResponse(url=f"{SSO_ISSUER}?error=token_failed", status_code=302)

    token_data = token_res.json()
    access_token = token_data.get("access_token")
    if not access_token:
        return RedirectResponse(url=f"{SSO_ISSUER}?error=no_token", status_code=302)

    redirect = RedirectResponse(url=return_to, status_code=302)
    redirect.set_cookie(
        key=ORCEST_SSO_COOKIE,
        value=access_token,
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=900,
    )
    return redirect


def _orchestration_html():
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Orcest AI Orchestration</title>
<style>*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:system-ui;background:#0a0a0f;color:#e0e0e0;min-height:100vh;padding:40px}}
.container{{max-width:800px;margin:0 auto}}
h1{{background:linear-gradient(135deg,#60a5fa,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:20px}}
.card{{background:#111827;border:1px solid #1f2937;border-radius:12px;padding:24px;margin-bottom:16px}}
.card h3{{color:#f8fafc;margin-bottom:8px}}
.card p{{color:#94a3b8}}
a{{color:#60a5fa;text-decoration:none}}
a:hover{{text-decoration:underline}}
.sso-status{{display:inline-flex;align-items:center;gap:8px;padding:8px 16px;background:#1e3a1e;color:#4ade80;border-radius:8px;margin-bottom:24px}}
.sso-status::before{{content:"✓";font-weight:bold}}
</style></head>
<body>
<div class="container">
<div class="sso-status">ورود با SSO انجام شد</div>
<h1>Orcest AI Orchestration</h1>
<p style="margin-bottom:24px;color:#94a3b8">سکو LangChain برای عامل‌های هوشمند. فورک از <a href="https://github.com/langchain-ai/langchain" target="_blank">LangChain</a>.</p>
<div class="card">
<h3>API Endpoints</h3>
<p><a href="/api/info">/api/info</a> – اطلاعات پلتفرم</p>
<p><a href="/health">/health</a> – وضعیت سرویس</p>
<p><a href="/ecosystem/health">/ecosystem/health</a> – سلامت اکوسیستم</p>
<p><a href="/api/langchain/ecosystem">/api/langchain/ecosystem</a> – اجزای کامل LangChain</p>
<p><a href="/api/langchain/deep-agents">/api/langchain/deep-agents</a> – دسترسی Deep Agents</p>
<p><a href="/api/langchain/langgraph">/api/langchain/langgraph</a> – دسترسی LangGraph</p>
<p><a href="/api/langchain/integrations">/api/langchain/integrations</a> – دسترسی Integrations</p>
<p><a href="/api/langchain/langsmith">/api/langchain/langsmith</a> – دسترسی LangSmith</p>
<p><a href="/api/langchain/langsmith-deployment">/api/langchain/langsmith-deployment</a> – دسترسی LangSmith Deployment</p>
<p><a href="/api/langchain/agent/run">/api/langchain/agent/run</a> – اجرای Agent روی API اورکست</p>
<p><a href="/orchestration/langchain">/orchestration/langchain</a> – UI اختصاصی LangChain (SSO)</p>
<p><a href="/orchestration/rainymodel">/orchestration/rainymodel</a> – UI اختصاصی RainyModel (SSO)</p>
</div>
<div class="card">
<h3>RainyModel API</h3>
<p>Base URL: <code>{RAINYMODEL_BASE_URL}</code></p>
<p>استفاده با کلید API از داشبورد SSO.</p>
<p>Manifest اتصال RainyModel به LangChain Orcest: <a href="/api/rainymodel/langchain-manifest">/api/rainymodel/langchain-manifest</a></p>
<p>Manifest Schema: <a href="/api/rainymodel/langchain-manifest/schema">/api/rainymodel/langchain-manifest/schema</a></p>
</div>
<p><a href="https://orcest.ai">← بازگشت به صفحه اصلی</a> | <a href="https://github.com/langchain-ai/langchain" target="_blank" rel="noopener">LangChain</a> | <a href="https://login.orcest.ai/logout">خروج</a></p>
</div></body></html>"""


def _langchain_console_html():
    return """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Orcest LangChain Console</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:system-ui;background:#0a0a0f;color:#e0e0e0;min-height:100vh;padding:28px}
.container{max-width:980px;margin:0 auto}
h1{background:linear-gradient(135deg,#60a5fa,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:10px}
p.sub{color:#94a3b8;margin-bottom:22px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px;margin-bottom:18px}
.card{background:#111827;border:1px solid #1f2937;border-radius:12px;padding:16px}
.card h3{color:#f8fafc;margin-bottom:8px;font-size:1rem}
.card p{color:#94a3b8;font-size:.92rem;margin-bottom:10px}
.row{display:flex;gap:8px;flex-wrap:wrap}
button,a.btn{background:#1f2937;color:#e2e8f0;border:1px solid #334155;border-radius:8px;padding:8px 12px;cursor:pointer;text-decoration:none;font-size:.86rem}
button:hover,a.btn:hover{border-color:#60a5fa;color:#60a5fa}
.panel{background:#0f172a;border:1px solid #1e293b;border-radius:12px;padding:14px}
pre{white-space:pre-wrap;word-break:break-word;color:#cbd5e1;font-size:.86rem}
.toplinks{margin-top:12px}
</style></head>
<body>
<div class="container">
<h1>Orcest LangChain Console</h1>
<p class="sub">Simple SSO-protected UI to access Deep Agents, LangGraph, Integrations, LangSmith and LangSmith Deployment through internal Orcest endpoints.</p>
<div class="grid">
  <div class="card"><h3>Deep Agents</h3><p>Planning, subagents, and filesystem-driven complex tasks.</p><div class="row"><button onclick="callApi('/api/langchain/deep-agents')">Open Endpoint</button></div></div>
  <div class="card"><h3>LangGraph</h3><p>Reliable orchestration with state, memory, and HITL patterns.</p><div class="row"><button onclick="callApi('/api/langchain/langgraph')">Open Endpoint</button></div></div>
  <div class="card"><h3>Integrations</h3><p>Providers, tools, toolkits and integration catalog access.</p><div class="row"><button onclick="callApi('/api/langchain/integrations')">Open Endpoint</button></div></div>
  <div class="card"><h3>LangSmith</h3><p>Observability, evals, tracing, and production debugging.</p><div class="row"><button onclick="callApi('/api/langchain/langsmith')">Open Endpoint</button></div></div>
  <div class="card"><h3>LangSmith Deployment</h3><p>Deploy and scale stateful long-running agent workflows.</p><div class="row"><button onclick="callApi('/api/langchain/langsmith-deployment')">Open Endpoint</button></div></div>
  <div class="card"><h3>RainyModel Access Map</h3><p>Per-component RainyModel integration and endpoint mapping.</p><div class="row"><button onclick="callApi('/api/rainymodel/langchain-manifest')">Open Manifest</button></div></div>
</div>
<div class="panel"><pre id="out">Select any component to load JSON response...</pre></div>
<div class="toplinks row">
  <a class="btn" href="/orchestration">Back to Orchestration</a>
  <a class="btn" href="/orchestration/rainymodel">RainyModel Console</a>
  <a class="btn" href="/api/rainymodel/langchain-manifest/schema">Manifest Schema</a>
  <a class="btn" href="https://github.com/langchain-ai/langchain" target="_blank" rel="noopener">LangChain Repo</a>
  <a class="btn" href="https://login.orcest.ai/logout">Logout</a>
</div>
</div>
<script>
async function callApi(path){
  const out = document.getElementById('out');
  out.textContent = 'Loading ' + path + ' ...';
  try {
    const res = await fetch(path, {headers: {'Accept':'application/json'}});
    const data = await res.json();
    out.textContent = JSON.stringify(data, null, 2);
  } catch (e) {
    out.textContent = 'Request failed: ' + e.message;
  }
}
</script>
</body></html>"""


def _rainymodel_console_html():
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Orcest RainyModel Console</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:system-ui;background:#0a0a0f;color:#e0e0e0;min-height:100vh;padding:28px}}
.container{{max-width:980px;margin:0 auto}}
h1{{background:linear-gradient(135deg,#60a5fa,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:10px}}
p.sub{{color:#94a3b8;margin-bottom:18px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px;margin-bottom:18px}}
.card{{background:#111827;border:1px solid #1f2937;border-radius:12px;padding:16px}}
.card h3{{color:#f8fafc;margin-bottom:8px;font-size:1rem}}
.card p{{color:#94a3b8;font-size:.92rem;margin-bottom:10px}}
.row{{display:flex;gap:8px;flex-wrap:wrap}}
button,a.btn{{background:#1f2937;color:#e2e8f0;border:1px solid #334155;border-radius:8px;padding:8px 12px;cursor:pointer;text-decoration:none;font-size:.86rem}}
button:hover,a.btn:hover{{border-color:#60a5fa;color:#60a5fa}}
.panel{{background:#0f172a;border:1px solid #1e293b;border-radius:12px;padding:14px}}
pre{{white-space:pre-wrap;word-break:break-word;color:#cbd5e1;font-size:.86rem}}
.toplinks{{margin-top:12px}}
.hint{{margin-top:10px;color:#94a3b8;font-size:.9rem}}
</style></head>
<body>
<div class="container">
<h1>Orcest RainyModel Console</h1>
<p class="sub">SSO dashboard for RainyModel access to Orcest LangChain APIs. Use per-component links or load manifest/schema.</p>
<div class="grid">
  <div class="card"><h3>Manifest</h3><p>Full RainyModel integration manifest with access map and examples.</p><div class="row"><button onclick="callApi('/api/rainymodel/langchain-manifest')">Open Manifest</button></div></div>
  <div class="card"><h3>Manifest Schema</h3><p>Schema to validate manifest payload before consumption.</p><div class="row"><button onclick="callApi('/api/rainymodel/langchain-manifest/schema')">Open Schema</button></div></div>
  <div class="card"><h3>Deep Agents Access</h3><p>RainyModel mapped endpoint for planning/subagent scenarios.</p><div class="row"><a class="btn" href="/api/langchain/deep-agents" target="_blank">Open Endpoint</a></div></div>
  <div class="card"><h3>LangGraph Access</h3><p>Mapped endpoint for stateful workflow orchestration.</p><div class="row"><a class="btn" href="/api/langchain/langgraph" target="_blank">Open Endpoint</a></div></div>
  <div class="card"><h3>LangSmith Access</h3><p>Mapped endpoint for observability and evals.</p><div class="row"><a class="btn" href="/api/langchain/langsmith" target="_blank">Open Endpoint</a></div></div>
  <div class="card"><h3>Execution Endpoint</h3><p>Unified POST entrypoint RainyModel can use upstream.</p><div class="row"><a class="btn" href="/api/langchain/agent/run" target="_blank">Open Endpoint</a></div></div>
</div>
<div class="panel"><pre id="out">Select any item to load JSON response...</pre></div>
<p class="hint">RainyModel Base URL: <code>{RAINYMODEL_BASE_URL}</code></p>
<div class="toplinks row">
  <a class="btn" href="/orchestration">Back to Orchestration</a>
  <a class="btn" href="/orchestration/langchain">LangChain Console</a>
  <a class="btn" href="/fc">System Diagram</a>
  <a class="btn" href="https://login.orcest.ai/logout">Logout</a>
</div>
</div>
<script>
async function callApi(path){{
  const out = document.getElementById('out');
  out.textContent = 'Loading ' + path + ' ...';
  try {{
    const res = await fetch(path, {{headers: {{'Accept':'application/json'}}}});
    const data = await res.json();
    out.textContent = JSON.stringify(data, null, 2);
  }} catch (e) {{
    out.textContent = 'Request failed: ' + e.message;
  }}
}}
</script>
</body></html>"""


def _fc_html():
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Orcest System Flowchart</title>
  <script src="https://cdn.jsdelivr.net/npm/@panzoom/panzoom@4.6.0/dist/panzoom.min.js"></script>
  <script type="module">
    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
    mermaid.initialize({ startOnLoad: true, theme: 'dark', securityLevel: 'loose' });

    function downloadFile(filename, content, mimeType) {
      const blob = new Blob([content], { type: mimeType });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    }

    function setupPanZoomAndExports() {
      const svg = document.querySelector('.mermaid svg');
      const host = document.getElementById('diagramHost');
      if (!svg || !host || !window.Panzoom) {
        return;
      }

      host.innerHTML = '';
      host.appendChild(svg);
      svg.style.width = '100%';
      svg.style.height = 'auto';

      const panzoom = window.Panzoom(svg, {
        maxScale: 3.5,
        minScale: 0.4,
        step: 0.2,
      });

      host.addEventListener('wheel', panzoom.zoomWithWheel, { passive: false });

      document.getElementById('zoomInBtn').addEventListener('click', () => panzoom.zoomIn());
      document.getElementById('zoomOutBtn').addEventListener('click', () => panzoom.zoomOut());
      document.getElementById('resetBtn').addEventListener('click', () => panzoom.reset());

      document.getElementById('exportSvgBtn').addEventListener('click', () => {
        const serializer = new XMLSerializer();
        const svgRaw = serializer.serializeToString(svg);
        downloadFile('orcest-system-diagram.svg', svgRaw, 'image/svg+xml;charset=utf-8');
      });

      document.getElementById('exportPngBtn').addEventListener('click', () => {
        const serializer = new XMLSerializer();
        const svgRaw = serializer.serializeToString(svg);
        const svg64 = btoa(unescape(encodeURIComponent(svgRaw)));
        const image64 = 'data:image/svg+xml;base64,' + svg64;
        const img = new Image();
        img.onload = () => {
          const canvas = document.createElement('canvas');
          canvas.width = img.width || 1600;
          canvas.height = img.height || 900;
          const ctx = canvas.getContext('2d');
          ctx.fillStyle = '#0a0a0f';
          ctx.fillRect(0, 0, canvas.width, canvas.height);
          ctx.drawImage(img, 0, 0);
          const pngUrl = canvas.toDataURL('image/png');
          const a = document.createElement('a');
          a.href = pngUrl;
          a.download = 'orcest-system-diagram.png';
          document.body.appendChild(a);
          a.click();
          a.remove();
        };
        img.src = image64;
      });
    }

    window.addEventListener('load', () => {
      setTimeout(setupPanZoomAndExports, 350);
    });
  </script>
  <style>
    body { margin: 0; background: #0a0a0f; color: #e2e8f0; font-family: system-ui, -apple-system, sans-serif; }
    .wrap { max-width: 1200px; margin: 0 auto; padding: 28px; }
    h1 { margin: 0 0 10px 0; font-size: 28px; }
    p { color: #94a3b8; margin-top: 0; }
    .card { margin-top: 18px; background: #111827; border: 1px solid #1f2937; border-radius: 14px; padding: 16px; }
    .toolbar { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 12px; }
    .toolbar button { background: #0f172a; color: #e2e8f0; border: 1px solid #334155; border-radius: 8px; padding: 8px 12px; cursor: pointer; }
    .toolbar button:hover { border-color: #60a5fa; color: #60a5fa; }
    #diagramHost { overflow: hidden; border: 1px dashed #334155; border-radius: 10px; min-height: 380px; display: flex; align-items: center; justify-content: center; padding: 8px; }
    #diagramHost svg { touch-action: none; cursor: grab; }
    .mermaid { display: none; }
    .links { margin-top: 14px; display: flex; gap: 10px; flex-wrap: wrap; }
    .links a { color: #60a5fa; text-decoration: none; border: 1px solid #334155; padding: 8px 12px; border-radius: 8px; }
    .links a:hover { border-color: #60a5fa; }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Orcest AI System Diagram</h1>
    <p>Unified architecture: Orcest core, SSO, LangChain APIs, RainyModel routing, and ecosystem services.</p>
    <div class="card">
      <div class="toolbar">
        <button id="zoomInBtn" type="button">Zoom In</button>
        <button id="zoomOutBtn" type="button">Zoom Out</button>
        <button id="resetBtn" type="button">Reset</button>
        <button id="exportSvgBtn" type="button">Export SVG</button>
        <button id="exportPngBtn" type="button">Export PNG</button>
      </div>
      <div id="diagramHost"></div>
      <pre class="mermaid">
flowchart TD
  userNode["User"]
  webNode["orcest.ai Web"]
  ssoNode["login.orcest.ai SSO"]
  orchestrationNode["Orchestration UI"]
  langchainUiNode["LangChain Console UI"]
  langchainApiNode["Orcest LangChain API"]
  manifestNode["RainyModel Manifest"]
  rainyModelNode["RainyModel Proxy"]
  providersNode["Model Providers"]

  deepAgentsNode["Deep Agents"]
  langGraphNode["LangGraph"]
  integrationsNode["Integrations"]
  langsmithNode["LangSmith"]
  deploymentNode["LangSmith Deployment"]

  laminoNode["llm.orcest.ai"]
  maestristNode["agent.orcest.ai"]
  orcideNode["ide.orcest.ai"]
  statusNode["status.orcest.ai"]

  userNode --> webNode
  webNode -->|"Login"| ssoNode
  ssoNode --> orchestrationNode
  orchestrationNode --> langchainUiNode

  langchainUiNode --> langchainApiNode
  langchainApiNode --> deepAgentsNode
  langchainApiNode --> langGraphNode
  langchainApiNode --> integrationsNode
  langchainApiNode --> langsmithNode
  langchainApiNode --> deploymentNode

  langchainApiNode --> manifestNode
  manifestNode --> rainyModelNode
  rainyModelNode --> providersNode

  webNode --> laminoNode
  webNode --> maestristNode
  webNode --> orcideNode
  webNode --> statusNode
      </pre>
    </div>
    <div class="links">
      <a href="/orchestration">Orchestration</a>
      <a href="/orchestration/langchain">LangChain Console</a>
      <a href="/api/langchain/ecosystem">LangChain Ecosystem API</a>
      <a href="/api/rainymodel/langchain-manifest">RainyModel Manifest</a>
    </div>
  </div>
</body>
</html>"""


@app.get("/orchestration", response_class=HTMLResponse)
async def orchestration_page(request: Request):
    """Orcest AI Orchestration - LangChain-based service (requires SSO)"""
    token = request.cookies.get(ORCEST_SSO_COOKIE) or request.headers.get("Authorization", "").replace("Bearer ", "")
    if await _is_authenticated_token(token):
        return HTMLResponse(content=_orchestration_html())
    return RedirectResponse(url=_auth_url_with_state("/orchestration"), status_code=302)


@app.get("/orchestration/langchain", response_class=HTMLResponse)
async def langchain_console_page(request: Request):
    """SSO-protected simple UI for internal LangChain component access."""
    token = request.cookies.get(ORCEST_SSO_COOKIE) or request.headers.get("Authorization", "").replace("Bearer ", "")
    if await _is_authenticated_token(token):
        return HTMLResponse(content=_langchain_console_html())
    return RedirectResponse(url=_auth_url_with_state("/orchestration/langchain"), status_code=302)


@app.get("/orchestration/rainymodel", response_class=HTMLResponse)
async def rainymodel_console_page(request: Request):
    """SSO-protected console for RainyModel manifest and access map usage."""
    token = request.cookies.get(ORCEST_SSO_COOKIE) or request.headers.get("Authorization", "").replace("Bearer ", "")
    if await _is_authenticated_token(token):
        return HTMLResponse(content=_rainymodel_console_html())
    return RedirectResponse(url=_auth_url_with_state("/orchestration/rainymodel"), status_code=302)


def _auth_url_with_state(return_to: str) -> str:
    state_obj = {"returnTo": return_to}
    state_raw = json.dumps(state_obj).encode("utf-8")
    encoded_state = base64.urlsafe_b64encode(state_raw).decode("utf-8").rstrip("=")
    return (
        f"{SSO_ISSUER}/oauth2/authorize?client_id={SSO_CLIENT_ID}"
        f"&redirect_uri={SSO_CALLBACK_URL}&response_type=code&scope=openid%20profile%20email"
        f"&state={encoded_state}"
    )


async def _is_authenticated_token(token: str) -> bool:
    if not token and not SSO_CLIENT_SECRET:
        return True
    if not token or not SSO_CLIENT_SECRET:
        return False
    async with httpx.AsyncClient() as client:
        verify_res = await client.post(
            f"{SSO_ISSUER}/api/token/verify",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        )
    return verify_res.status_code == 200 and verify_res.json().get("valid")


@app.get("/fc", response_class=HTMLResponse)
async def flowchart_page():
    """Public flowchart page for system architecture."""
    return HTMLResponse(content=_fc_html())
