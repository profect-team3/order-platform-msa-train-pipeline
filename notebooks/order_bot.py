import requests
import time
import random
import json
from datetime import datetime, timedelta

# Configuration
BASE_URL = "http://localhost:8080"
ORDER_URL = f"{BASE_URL}/order/order"
SIGNUP_URL = f"{BASE_URL}/user/user/signup"
LOGIN_URL = f"{BASE_URL}/auth/auth/login"
ADD_TO_CART_URL = f"{BASE_URL}/order/order/item"

# User credentials for signup and login
SIGNUP_PAYLOAD = {
    "username": "user1", 
    "password": "user1passwd", 
    "email": "user1@example.com", 
    "nickname": "user1", 
    "realName": "홍길동", 
    "phoneNumber": "01053461367", 
    "userRole": "CUSTOMER"
}

LOGIN_PAYLOAD = {
    "username": "user1",
    "password": "user1passwd"
}

# Payload for adding item to cart (once)
ADD_TO_CART_PAYLOAD = {
    "menuId" : "93035343-e924-4d97-8561-90a3ebdfe355",
    "storeId" : "e0b0fa7e-08a0-4c87-9111-7057d83c1fc8",
    "quantity" : 1
}

# Base order payload (totalPrice will be dynamic)
BASE_ORDER_PAYLOAD = {
  "paymentMethod" : "CREDIT_CARD",
  "orderChannel" : "ONLINE",
  "receiptMethod" : "DELIVERY",
  "requestMessage" : "특별한 요청사항 없음",
  "totalPrice" : 0, # This will be set dynamically
  "deliveryAddress" : "서울시 마포구 월드컵로 250"
}

# Constants for order generation logic
AVG_PRICE_PER_ORDER = 15000
TOTAL_SIMULATED_HOURS = 24 * 7 # Simulate 7 days of data into the future

def send_signup_request():
    """Sends a single user signup request."""
    print(f"Attempting to sign up user to {SIGNUP_URL}")
    try:
        response = requests.post(SIGNUP_URL, headers={"Content-Type": "application/json"}, data=json.dumps(SIGNUP_PAYLOAD))
        response.raise_for_status()  # Raise an exception for HTTP errors
        print(f"Signup successful! Status Code: {response.status_code}, Response: {response.json()}")
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 409: # Conflict, user already exists
            print(f"Signup skipped: User '{SIGNUP_PAYLOAD['username']}' already exists (Status Code: 409).")
        else:
            print(f"Error during signup: {e}")
    except requests.exceptions.RequestException as e:
        print(f"Error during signup: {e}")

def send_login_request():
    """Sends a single user login request and returns the session object and bearer token."""
    print(f"Attempting to log in user to {LOGIN_URL}")
    session = requests.Session() # Use a session to persist cookies/auth
    bearer_token = None
    try:
        response = session.post(LOGIN_URL, headers={"Content-Type": "application/json"}, data=json.dumps(LOGIN_PAYLOAD))
        response.raise_for_status()  # Raise an exception for HTTP errors
        login_response_json = response.json()
        print(f"Login successful! Status Code: {response.status_code}, Response: {login_response_json}")
        
        # Extract the token from 'accessToken' field within the 'result' object
        if 'result' in login_response_json and 'accessToken' in login_response_json['result']:
            bearer_token = login_response_json['result']['accessToken']
            print(f"Extracted Bearer Token: {bearer_token[:10]}...") # Print first 10 chars for brevity
        else:
            print("Warning: No 'accessToken' found in login response 'result' object.")

        return session, bearer_token
    except requests.exceptions.RequestException as e:
        print(f"Error during login: {e}")
        return None, None

def send_add_to_cart_request(session, token):
    """Sends a single request to add an item to the cart with bearer token."""
    print(f"Attempting to add item to cart at {ADD_TO_CART_URL}")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        response = session.post(ADD_TO_CART_URL, headers=headers, data=json.dumps(ADD_TO_CART_PAYLOAD))
        response.raise_for_status() # Raise an exception for HTTP errors
        print(f"Add to cart successful! Status Code: {response.status_code}, Response: {response.json()}")
    except requests.exceptions.RequestException as e:
        print(f"Error adding item to cart: {e}")

def send_single_order(session, token, total_price):
    """Sends a single order creation request with a dynamic total_price and bearer token."""
    order_payload = BASE_ORDER_PAYLOAD.copy()
    order_payload["totalPrice"] = total_price
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        response = session.post(ORDER_URL, headers=headers, data=json.dumps(order_payload))
        response.raise_for_status()  # Raise an exception for HTTP errors
        print(f"Order sent (Price: {total_price})! Status Code: {response.status_code}, Response: {response.json()}")
    except requests.exceptions.RequestException as e:
        print(f"Error sending order (Price: {total_price}): {e}")

def run_order_bot():
    """Runs the order bot, performing signup, login, add to cart, and then sending orders based on time-weighted logic."""
    send_signup_request()
    session, bearer_token = send_login_request()

    if session and bearer_token:
        send_add_to_cart_request(session, bearer_token)
        print(f"Starting order bot to simulate {TOTAL_SIMULATED_HOURS} hours of orders into the future to {ORDER_URL}")
        now = datetime.now()

        for i in range(TOTAL_SIMULATED_HOURS):
            # Simulate an hour from the current time onwards
            timestamp = now + timedelta(hours=i)
            hour = timestamp.hour

            # Determine base order quantity based on time of day
            base_quantity = 0
            if 11 <= hour <= 13:  # Lunch peak
                base_quantity = 20
            elif 17 <= hour <= 20: # Dinner peak
                base_quantity = 30
            else: # Off-peak
                base_quantity = 5
            
            # Add fluctuation to quantity (-2 to +2)
            real_quantity = base_quantity + random.randint(-2, 2)
            real_quantity = max(0, real_quantity) # Ensure non-negative

            print(f"\n--- Simulating hour {timestamp.strftime('%Y-%m-%d %H:00')} (Quantity: {real_quantity}) ---")

            for _ in range(real_quantity):
                # Price fluctuation (-1000 to +1000)
                price_fluctuation = random.randint(-1000, 1000)
                dynamic_total_price = AVG_PRICE_PER_ORDER + price_fluctuation
                dynamic_total_price = max(1000, dynamic_total_price) # Ensure minimum price

                send_single_order(session, bearer_token, dynamic_total_price)
                # Small delay between individual orders within the same hour
                time.sleep(random.uniform(0.1, 0.5))
            
            # Longer delay between simulated hours
            time.sleep(random.uniform(1, 3)) 

        print("Order bot finished simulating all hours.")
    else:
        print("Login failed or no bearer token found, cannot proceed with order requests.")

if __name__ == "__main__":
    run_order_bot()
