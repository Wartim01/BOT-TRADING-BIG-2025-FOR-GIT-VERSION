import hashlib
import hmac
import time
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class AuthManager:
    def __init__(self, secret):
        self.secret = secret
    def generate_token(self, user_id):
        data = f"{user_id}:{int(time.time())}"
        token = hmac.new(self.secret.encode(), data.encode(), hashlib.sha256).hexdigest()
        logger.info("Generated token for user %s", user_id)
        return token
    def verify_token(self, token, user_id, timestamp, tolerance=300):
        data = f"{user_id}:{timestamp}"
        expected = hmac.new(self.secret.encode(), data.encode(), hashlib.sha256).hexdigest()
        if expected != token:
            logger.warning("Token verification failed for user %s", user_id)
            return False
        if abs(int(time.time()) - int(timestamp)) > tolerance:
            logger.warning("Token expired for user %s", user_id)
            return False
        logger.info("Token verified for user %s", user_id)
        return True
