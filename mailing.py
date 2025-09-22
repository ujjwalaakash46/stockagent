import os
from dotenv import load_dotenv
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
import smtplib
import ssl

load_dotenv()

SENDER_EMAIL = os.getenv('SENDER_EMAIL') # ✨ ADDED
GMAIL_APP_PASSWORD = os.getenv('GMAIL_APP_PASSWORD') # ✨ ADDED
RECIPIENT_EMAIL = os.getenv('RECIPIENT_EMAIL') # ✨ ADDED

class StockAIEmailNotifier:
    """Handles sending email notifications for stock analysis results."""
    def __init__(self, smtp_server="smtp.gmail.com", port=587):
        self.smtp_server = smtp_server
        self.port = port
        self.sender_email = SENDER_EMAIL
        self.sender_password = GMAIL_APP_PASSWORD
    
    def format_email_body(self, analysis_result, symbol):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")
        content = str(analysis_result.get('analysis_result', 'No detailed analysis found.'))
        
        subject = f"🤖 StockAI Analysis Report - {symbol} - {datetime.now().strftime('%Y-%m-%d')}"
        
        html_content = f"""
        <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; }}
                    .header {{ background-color: #2c3e50; color: white; padding: 20px; text-align: center; }}
                    .content {{ padding: 20px; }}
                    .analysis-section {{ background-color: #f8f9fa; border-left: 4px solid #3498db; padding: 15px; margin-top: 20px; }}
                    .footer {{ background-color: #ecf0f1; padding: 10px; text-align: center; font-size: 12px; color: #7f8c8d; }}
                </style>
            </head>
            <body>
                <div class="header"><h1>StockAI Analysis Report: {symbol}</h1></div>
                <div class="content">
                    <p><strong>Generated on:</strong> {timestamp}</p>
                    <div class="analysis-section">
                        <h2>Final Recommendation & Analysis</h2>
                        <pre style="white-space: pre-wrap; font-family: 'Courier New', monospace;">{content}</pre>
                    </div>
                </div>
            </body>
        </html>
        """
        return subject, html_content

    def send_email(self, analysis_result, symbol):
        if not self.sender_email or not self.sender_password:
            print("❌ Email credentials not configured. Please set SENDER_EMAIL and GMAIL_APP_PASSWORD in .env file.")
            return False
            
        subject, html_body = self.format_email_body(analysis_result, symbol)
        
        message = MIMEMultipart()
        message["From"] = self.sender_email
        message["To"] = RECIPIENT_EMAIL
        message["Subject"] = subject
        message.attach(MIMEText(html_body, "html"))
        
        try:
            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_server, self.port) as server:
                server.starttls(context=context)
                server.login(self.sender_email, self.sender_password)
                server.sendmail(self.sender_email, RECIPIENT_EMAIL, message.as_string())
            print(f"✅ Email notification sent successfully to {RECIPIENT_EMAIL}")
            return True
        except Exception as e:
            print(f"❌ Failed to send email: {str(e)}")
            return False