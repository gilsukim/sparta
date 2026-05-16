#!/usr/bin/env python3
"""Streamlit web UI for the multi-LLM aggregator."""

import asyncio

import streamlit as st

from multi_llm import query_all

st.set_page_config(page_title="Multi-LLM", page_icon="🤖", layout="wide")
st.title("Multi-LLM Prompt Runner")
st.caption("Sends your prompt to Claude, GPT-4o, and Gemini in parallel.")

prompt = st.text_area("Prompt", placeholder="Ask anything…", height=120)
run = st.button("Query all LLMs", type="primary", disabled=not prompt.strip())

if run and prompt.strip():
    with st.spinner("Querying all LLMs in parallel…"):
        results = asyncio.run(query_all(prompt.strip()))

    cols = st.columns(len(results))
    for col, result in zip(cols, results):
        with col:
            st.subheader(result.provider)
            st.caption(result.model)
            if result.error:
                st.error(result.error)
            else:
                st.markdown(result.response or "*(empty response)*")

    ok = sum(1 for r in results if r.error is None)
    st.divider()
    st.caption(f"{ok}/{len(results)} LLMs responded successfully")
