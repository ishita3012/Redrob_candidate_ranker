"""
Streamlit Demo App - Intelligent Candidate Ranker

This is the sandbox/demo required for submission.
Run with: streamlit run app.py

Features:
- Upload candidate JSON/JSONL files
- See rankings with explanations
- Explore individual candidate scores
- Download results as CSV
"""

import streamlit as st
import json
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_loader import load_candidates
from ranker import rank_candidates, RankedCandidate
from honeypot_detector import detect_honeypot, get_honeypot_score
from disqualifiers import detect_disqualifiers
from scorers import compute_all_scores
from config import SCORING_WEIGHTS, JD_REQUIREMENTS

# Page config
st.set_page_config(
    page_title="Intelligent Candidate Ranker",
    page_icon="🎯",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .score-high { color: #28a745; font-weight: bold; }
    .score-medium { color: #ffc107; font-weight: bold; }
    .score-low { color: #dc3545; font-weight: bold; }
    .metric-card {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)


def load_sample_data():
    """Load sample candidates for demo."""
    sample_path = Path(__file__).parent.parent / 'sample_candidates.json'
    if sample_path.exists():
        with open(sample_path) as f:
            return json.load(f)
    return None


def parse_uploaded_file(uploaded_file):
    """Parse uploaded JSON or JSONL file."""
    content = uploaded_file.read().decode('utf-8')

    # Try JSONL first
    candidates = []
    for line in content.strip().split('\n'):
        if line.strip():
            try:
                candidates.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    if candidates:
        return candidates

    # Try regular JSON
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return data
        return [data]
    except json.JSONDecodeError:
        return None


def get_score_class(score):
    """Get CSS class based on score."""
    if score >= 0.6:
        return "score-high"
    elif score >= 0.4:
        return "score-medium"
    return "score-low"


def main():
    st.title("🎯 Intelligent Candidate Ranker")
    st.markdown("### Redrob Hackathon - Data & AI Challenge")

    st.markdown("""
    This tool ranks candidates for a **Senior AI Engineer** role using multi-dimensional scoring:
    - **Beyond keyword matching** - analyzes career trajectory, skill trust, behavioral signals
    - **Honeypot detection** - catches impossible profiles
    - **Disqualifier detection** - identifies JD red flags
    - **Explainable reasoning** - every ranking has justification
    """)

    st.divider()

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        st.subheader("Scoring Weights")
        st.caption("Adjust how much each dimension matters")

        # Display current weights
        for dim, weight in SCORING_WEIGHTS.items():
            st.text(f"{dim}: {weight:.0%}")

        st.divider()

        st.subheader("JD Requirements")
        exp_range = JD_REQUIREMENTS['experience_range']
        st.text(f"Experience: {exp_range[0]}-{exp_range[1]} years")
        st.text(f"Notice: <{JD_REQUIREMENTS['preferred_notice_days']}d preferred")

    # Main content
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📤 Upload Candidates")

        # File upload
        uploaded_file = st.file_uploader(
            "Upload JSON or JSONL file",
            type=['json', 'jsonl'],
            help="Upload candidates in JSON or JSONL format"
        )

        # Or use sample data
        use_sample = st.checkbox("Use sample data (50 candidates)")

        candidates = None

        if uploaded_file:
            candidates = parse_uploaded_file(uploaded_file)
            if candidates:
                st.success(f"Loaded {len(candidates)} candidates")
            else:
                st.error("Failed to parse file")
        elif use_sample:
            candidates = load_sample_data()
            if candidates:
                st.success(f"Loaded {len(candidates)} sample candidates")
            else:
                st.warning("Sample data not found")

        # Rank button
        if candidates and st.button("🚀 Rank Candidates", type="primary"):
            with st.spinner("Ranking candidates..."):
                ranked = rank_candidates(candidates, top_n=min(100, len(candidates)))
                st.session_state['ranked'] = ranked
                st.session_state['candidates'] = {c['candidate_id']: c for c in candidates}

    with col2:
        st.subheader("📊 Results")

        if 'ranked' in st.session_state:
            ranked = st.session_state['ranked']

            # Summary metrics
            metrics_cols = st.columns(4)
            with metrics_cols[0]:
                st.metric("Total Ranked", len(ranked))
            with metrics_cols[1]:
                honeypots = sum(1 for r in ranked if r.is_honeypot)
                st.metric("Honeypots", honeypots)
            with metrics_cols[2]:
                avg_score = sum(r.score for r in ranked) / len(ranked)
                st.metric("Avg Score", f"{avg_score:.3f}")
            with metrics_cols[3]:
                top_score = ranked[0].score if ranked else 0
                st.metric("Top Score", f"{top_score:.3f}")

            st.divider()

            # Rankings table
            df = pd.DataFrame([
                {
                    'Rank': r.rank,
                    'Candidate': r.candidate_id,
                    'Score': f"{r.score:.4f}",
                    'Reasoning': r.reasoning[:100] + "..." if len(r.reasoning) > 100 else r.reasoning
                }
                for r in ranked
            ])

            st.dataframe(df, use_container_width=True, height=400)

            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                csv,
                "ranking_results.csv",
                "text/csv"
            )
        else:
            st.info("Upload candidates and click 'Rank Candidates' to see results")

    # Detailed view
    st.divider()
    st.subheader("🔍 Candidate Deep Dive")

    if 'ranked' in st.session_state and 'candidates' in st.session_state:
        ranked = st.session_state['ranked']
        candidates_map = st.session_state['candidates']

        # Select candidate
        selected_id = st.selectbox(
            "Select a candidate to analyze",
            options=[r.candidate_id for r in ranked],
            format_func=lambda x: f"#{next(r.rank for r in ranked if r.candidate_id == x)} - {x}"
        )

        if selected_id:
            candidate = candidates_map[selected_id]
            ranked_info = next(r for r in ranked if r.candidate_id == selected_id)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Profile")
                profile = candidate.get('profile', {})
                st.markdown(f"**Title:** {profile.get('current_title', 'N/A')}")
                st.markdown(f"**Experience:** {profile.get('years_of_experience', 0):.1f} years")
                st.markdown(f"**Location:** {profile.get('location', 'N/A')}")
                st.markdown(f"**Company:** {profile.get('current_company', 'N/A')}")

                st.markdown("#### Summary")
                st.text(profile.get('summary', 'N/A')[:500])

            with col2:
                st.markdown("#### Scores")

                # Compute detailed scores
                is_hp, hp_reasons = detect_honeypot(candidate)
                dq, dq_penalty = detect_disqualifiers(candidate)
                all_scores = compute_all_scores(candidate)

                for dim, (score, reasons) in all_scores.items():
                    score_class = get_score_class(score)
                    weight = SCORING_WEIGHTS.get(dim, 0)
                    st.markdown(f"""
                    <div class="metric-card">
                        <strong>{dim.replace('_', ' ').title()}</strong>
                        (weight: {weight:.0%})<br>
                        Score: <span class="{score_class}">{score:.3f}</span><br>
                        <small>{'; '.join(reasons[:2]) if reasons else 'N/A'}</small>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("#### Flags")
                if is_hp:
                    st.error(f"🚨 HONEYPOT: {hp_reasons}")
                if dq:
                    st.warning(f"⚠️ Disqualifiers: {', '.join(dq)}")
                if not is_hp and not dq:
                    st.success("✅ No red flags")

            # Behavioral signals
            st.markdown("#### Behavioral Signals")
            signals = candidate.get('redrob_signals', {})

            sig_cols = st.columns(5)
            with sig_cols[0]:
                st.metric("Response Rate", f"{signals.get('recruiter_response_rate', 0):.0%}")
            with sig_cols[1]:
                st.metric("Notice Period", f"{signals.get('notice_period_days', 'N/A')}d")
            with sig_cols[2]:
                st.metric("Open to Work", "Yes" if signals.get('open_to_work_flag') else "No")
            with sig_cols[3]:
                st.metric("GitHub Score", f"{signals.get('github_activity_score', -1):.0f}")
            with sig_cols[4]:
                st.metric("Interview Rate", f"{signals.get('interview_completion_rate', 0):.0%}")

            # Skills
            st.markdown("#### Top Skills")
            skills = candidate.get('skills', [])[:10]
            skills_df = pd.DataFrame([
                {
                    'Skill': s['name'],
                    'Proficiency': s.get('proficiency', 'N/A'),
                    'Duration': f"{s.get('duration_months', 0)} mo",
                    'Endorsements': s.get('endorsements', 0)
                }
                for s in skills
            ])
            st.dataframe(skills_df, use_container_width=True)

            # Reasoning
            st.markdown("#### Final Reasoning")
            st.info(ranked_info.reasoning)

    # Footer
    st.divider()
    st.caption("Built for Redrob Hackathon - Intelligent Candidate Discovery & Ranking Challenge")


if __name__ == '__main__':
    main()
