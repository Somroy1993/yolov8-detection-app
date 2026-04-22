"""Database and email operations for NeonDB."""
import os
import psycopg2
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

def insert_user_email(email: str, tool_name: str) -> None:
    """Insert user email into NeonDB."""
    try:
        conn = psycopg2.connect(os.getenv("NEON_DATABASE_URL"))
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO user_emails (email, tool, created_at) VALUES (%s, %s, %s)",
            (email, tool_name, datetime.now())
        )
        
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        raise Exception(f"Database error: {str(e)}")

def send_notification_email(user_email: str, tool_name: str) -> None:
    """Send notification email to admin."""
    try:
        smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        smtp_user = os.getenv("SMTP_USER")
        smtp_pass = os.getenv("SMTP_PASS")
        notify_email = os.getenv("NOTIFY_EMAIL", "somroymail@gmail.com")
        
        server = smtplib.SMTP(smtp_host, smtp_port)
        server.starttls()
        server.login(smtp_user, smtp_pass)
        
        msg = MIMEMultipart()
        msg["From"] = smtp_user
        msg["To"] = notify_email
        msg["Subject"] = f"[New Lead] {tool_name} - {user_email}"
        
        body = f"""
        New user registration:
        
        Tool: {tool_name}
        Email: {user_email}
        Timestamp: {datetime.now()}
        """
        
        msg.attach(MIMEText(body, "plain"))
        server.send_message(msg)
        server.quit()
    except Exception as e:
        raise Exception(f"Email error: {str(e)}")
