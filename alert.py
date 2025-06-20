import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email_alert(subject, body, receiver_email):
    # Sender email credentials
    sender_email = "your_email@gmail.com"
    sender_password = "your_app_password"  # Use App Password if using Gmail with 2FA

    # Create the email message
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    try:
        # Setup SMTP server (Gmail example)
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        text = msg.as_string()
        server.sendmail(sender_email, receiver_email, text)
        server.quit()
        print(f"[ALERT SENT] Email sent to {receiver_email}")
    except Exception as e:
        print(f"[ERROR] Failed to send alert: {e}")
