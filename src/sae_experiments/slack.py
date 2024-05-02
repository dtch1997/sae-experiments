import requests

def send_slack_notification(
    message: str,
    webhook_url: str,
    user_id: str,
):
    text = f"<@{user_id}> {message}"
    response = requests.post(
        webhook_url,
        json={"text": text},
        headers={'Content-type': 'application/json'}
    )
    response.raise_for_status()
