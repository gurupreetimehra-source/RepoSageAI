
import streamlit as st
import requests

st.set_page_config(
    page_title="RepoSage AI",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 RepoSage AI")
st.subheader("Your AI-powered open-source contribution mentor")
st.write(
    "Understand GitHub issues, discover relevant code, "
    "and get a beginner-friendly plan to contribute."
)

issue_url = st.text_input(
    "Paste a public GitHub issue URL",
    placeholder="https://github.com/owner/repo/issues/123"
)

if st.button("Analyze Issue", type="primary"):
    if not issue_url.startswith("https://github.com/"):
        st.error("Please enter a valid GitHub issue URL.")
    else:
        try:
            clean_url = issue_url.split("?")[0].rstrip("/")
            parts = clean_url.split("/")

            if len(parts) < 7 or parts[5] != "issues":
                st.error(
                    "Use a URL like "
                    "https://github.com/owner/repo/issues/123"
                )
            else:
                owner = parts[3]
                repo = parts[4]
                issue_number = parts[6]

                if not issue_number.isdigit():
                    st.error("The issue number must be numeric.")
                else:
                    api_url = (
                        f"https://api.github.com/repos/"
                        f"{owner}/{repo}/issues/{issue_number}"
                    )

                    response = requests.get(
                        api_url,
                        timeout=15,
                        headers={
                            "Accept": "application/vnd.github+json"
                        }
                    )

                    if response.status_code == 200:
                        issue = response.json()

                        st.success("GitHub issue loaded successfully!")

                        st.markdown(f"## {issue['title']}")
                        st.write(
                            f"**Repository:** {owner}/{repo}"
                        )
                        st.write(
                            f"**Status:** {issue['state']}"
                        )
                        st.write(
                            f"**Author:** {issue['user']['login']}"
                        )
                        st.markdown("### Issue description")

                        description = issue.get("body")
                        st.write(
                            description
                            if description
                            else "No description was provided."
                        )

                        st.link_button(
                            "Open original issue",
                            issue["html_url"]
                        )

                    elif response.status_code == 404:
                        st.error(
                            "Issue not found. Check the URL "
                            "or make sure the repository is public."
                        )
                    else:
                        st.error(
                            f"GitHub returned status "
                            f"{response.status_code}. Try again."
                        )

        except requests.RequestException:
            st.error(
                "Could not connect to GitHub. "
                "Check your internet connection and try again."
            )
