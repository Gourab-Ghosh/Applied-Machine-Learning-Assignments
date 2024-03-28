# test.py
import unittest
import requests
import os
import subprocess
import signal
import time

HAM_MESSAGES = [
    r"Just a friendly reminder that the deadline for our current project is next Friday, March 24th. Please make sure all your work is submitted by 5:00 PM on that day.",
    r"This is to confirm our meeting scheduled for tomorrow, March 13th, at 10:00 AM in the conference room. We'll be discussing the new marketing strategy.",
    r"Welcome to this week's edition of our newsletter! Inside, you'll find updates on recent company achievements, upcoming events, and employee spotlights.",
    r"We value your feedback! Please take a moment to complete our customer service survey and let us know how we're doing.",
    r"Please note that our offices will be closed on April 7th in observance of Good Friday. Regular business hours will resume on Monday, April 10th.",
    r"We appreciate your recent purchase and hope you're satisfied with your new product. If you have any questions or concerns, please don't hesitate to contact our support team.",
    r"Join us for an informative webinar on the latest trends in digital marketing, scheduled for March 20th at 2:00 PM. Register now to secure your spot!",
    r"Attached is the monthly performance report for your review. Please take a look and let me know if you have any questions or require further clarification.",
    r"Mark your calendars for our upcoming Staff Appreciation Day on April 14th! We have a fun-filled day planned to celebrate our amazing team.",
    r"Please be advised that our IT department will be performing system maintenance this Saturday, March 18th, from 8:00 AM to 12:00 PM. During this time, access to certain online services may be temporarily unavailable.",
]

SPAM_MESSAGES = [
    r"Congratulations! You've been selected to receive a free vacation to the Bahamas. Click here to claim your prize now!",
    r"Limited time offer! Get 50% off on all electronics. Hurry, sale ends soon. Visit our website to shop now.",
    r"You have won $1,000,000 in our lottery! To claim your prize, please send us your personal details and bank information.",
    r"Attention! Your computer has been infected with a virus. Download our antivirus software immediately to protect your device.",
    r"Exclusive deal for you! Buy one, get one free on all our products. Don't miss out on this amazing offer.",
    r"Warning! Your account will be suspended if you do not update your payment information within 24 hours. Click here to update now.",
    r"You're invited to join our exclusive investment club. Invest with us and earn guaranteed high returns. Contact us for more details.",
    r"Flash sale! All items must go. Up to 70% off on selected products. Shop now before it's too late.",
    r"Your email has been randomly selected to receive a $500 gift card. Click here to claim your reward.",
    r"Special announcement! We're giving away free samples of our new product. Sign up now to receive yours.",
]

class TestDockerApp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Generate a Docker container
        os.system("sudo docker build -t flask-app:latest .")
        # Start the Docker container in a separate process
        cls.docker_process = subprocess.Popen(["sudo", "docker", "run", "-p", "8576:8576", "--name", "flask-app", "flask-app:latest"])
        # Wait for the Docker container to start
        time.sleep(3)

    @classmethod
    def tearDownClass(cls):
        # Terminate the Docker container process by sending a SIGINT signal to the subprocess
        os.kill(cls.docker_process.pid, signal.SIGINT)
        time.sleep(3)
        os.system("sudo docker container rm flask-app")
        os.system("sudo docker image rm flask-app:latest --force")

    def test_docker_app_invalid_response(self):
        """Test the Flask endpoint through Docker."""
        response = requests.post("http://0.0.0.0:8576/score", json={})
        self.assertEqual(response.status_code, 400)

    for i, message in enumerate(HAM_MESSAGES):
        exec(f"test_ham_score_{i} = lambda self: docker_tester(self, \"{message}\", 0)")
    for i, message in enumerate(SPAM_MESSAGES):
        exec(f"test_spam_score_{i} = lambda self: docker_tester(self, \"{message}\", 1)")

def docker_tester(docker_unittest_instance: TestDockerApp, message: str, expected_value):
    """Test the Flask endpoint through Docker."""
    response = requests.post("http://0.0.0.0:8576/score", json={"text": message, "threshold": 0.5})
    response_json = response.json()
    docker_unittest_instance.assertEqual(response.status_code, 200)
    docker_unittest_instance.assertIn("prediction", response_json)
    docker_unittest_instance.assertIn("propensity", response_json)
    if "prediction" in response_json:
        docker_unittest_instance.assertEqual(response_json["prediction"], expected_value)
    if "propensity" in response_json:
        propensity = response_json["propensity"]
        docker_unittest_instance.assertGreaterEqual(propensity, 0)
        docker_unittest_instance.assertLessEqual(propensity, 1)

if __name__ == "__main__":
    unittest.main()
