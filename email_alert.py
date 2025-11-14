import smtplib

def send_alert():
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login("your_email@gmail.com", "your_password")
    message = "Subject: ALERT! Violence Detected\n\nImmediate action required."
    server.sendmail("your_email@gmail.com", "receiver_email@gmail.com", message)
    server.quit()
