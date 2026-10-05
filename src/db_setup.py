import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "banking_fraud.db")

def init_db(db_path=DB_PATH):
    """Initializes SQLite Database with schema and sample data."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Schema creation
    cursor.executescript("""
    CREATE TABLE IF NOT EXISTS Customer (
        customer_id INTEGER PRIMARY KEY,
        first_name VARCHAR(50),
        last_name VARCHAR(40),
        email VARCHAR(100),
        phone VARCHAR(20),
        date_of_birth DATE,
        address VARCHAR(250),
        customer_status VARCHAR(50)
    );

    CREATE TABLE IF NOT EXISTS Branch (
        branch_id INTEGER PRIMARY KEY,
        branch_name VARCHAR(20),
        city VARCHAR(25),
        state VARCHAR(15),
        ifsc_code VARCHAR(30) UNIQUE
    );

    CREATE TABLE IF NOT EXISTS accounts_details (
        account_id INTEGER PRIMARY KEY,
        account_type VARCHAR(20),
        account_number VARCHAR(30) UNIQUE,
        balance DECIMAL(15, 2),
        open_date DATE,
        account_status VARCHAR(20),
        customer_id INTEGER,
        branch_id INTEGER,
        FOREIGN KEY (customer_id) REFERENCES Customer(customer_id),
        FOREIGN KEY (branch_id) REFERENCES Branch(branch_id)
    );

    CREATE TABLE IF NOT EXISTS Accounts_type (
        account_type_id INTEGER PRIMARY KEY,
        type_name VARCHAR(20),
        minimum_balance DECIMAL(15, 2),
        interest DECIMAL(15, 2)
    );

    CREATE TABLE IF NOT EXISTS Beneficiary (
        beneficiary_id INTEGER PRIMARY KEY,
        beneficiary_name VARCHAR(20),
        beneficiary_account VARCHAR(30),
        bank_name VARCHAR(40),
        ifsc_code VARCHAR(30),
        added_date DATE,
        account_id INTEGER,
        FOREIGN KEY (account_id) REFERENCES accounts_details(account_id)
    );

    CREATE TABLE IF NOT EXISTS Transaction_type (
        transaction_type_id INTEGER PRIMARY KEY,
        type_name VARCHAR(20),
        description VARCHAR(250)
    );

    CREATE TABLE IF NOT EXISTS Transaction_record (
        transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
        amount DECIMAL(15, 2),
        transaction_date DATETIME,
        transaction_status VARCHAR(50),
        reference_number VARCHAR(40),
        description VARCHAR(200),
        account_id INTEGER,
        beneficiary_id INTEGER,
        transaction_type_id INTEGER,
        FOREIGN KEY (account_id) REFERENCES accounts_details(account_id),
        FOREIGN KEY (beneficiary_id) REFERENCES Beneficiary(beneficiary_id),
        FOREIGN KEY (transaction_type_id) REFERENCES Transaction_type(transaction_type_id)
    );

    CREATE TABLE IF NOT EXISTS Transaction_location (
        location_id INTEGER PRIMARY KEY AUTOINCREMENT,
        city VARCHAR(50),
        state VARCHAR(50),
        country VARCHAR(50),
        latitude DECIMAL(10, 7),
        longitude DECIMAL(10, 7),
        transaction_id INTEGER,
        FOREIGN KEY (transaction_id) REFERENCES Transaction_record(transaction_id)
    );

    CREATE TABLE IF NOT EXISTS Device (
        device_id INTEGER PRIMARY KEY,
        device_type VARCHAR(40),
        device_model VARCHAR(80),
        operating_system VARCHAR(50),
        device_identifier VARCHAR(15) UNIQUE,
        registered_date DATE,
        customer_id INTEGER,
        FOREIGN KEY (customer_id) REFERENCES Customer(customer_id)
    );

    CREATE TABLE IF NOT EXISTS Login_activitys (
        login_id INTEGER PRIMARY KEY AUTOINCREMENT,
        ip_address VARCHAR(40),
        login_time DATETIME,
        city VARCHAR(50),
        latitude DECIMAL(10, 7),
        longitude DECIMAL(10, 7),
        login_status VARCHAR(50),
        customer_id INTEGER,
        device_id INTEGER,
        FOREIGN KEY (customer_id) REFERENCES Customer(customer_id),
        FOREIGN KEY (device_id) REFERENCES Device(device_id)
    );

    CREATE TABLE IF NOT EXISTS Fraud_rule (
        rule_id INTEGER PRIMARY KEY,
        rule_name VARCHAR(50),
        rule_description VARCHAR(200),
        risk_level VARCHAR(50),
        threshhold_value DECIMAL(15, 2),
        rule_status VARCHAR(50)
    );

    CREATE TABLE IF NOT EXISTS Suspicious_transaction (
        suspicious_id INTEGER PRIMARY KEY AUTOINCREMENT,
        risk_score INTEGER,
        detected_date DATETIME,
        status VARCHAR(20),
        transaction_id INTEGER,
        rule_id INTEGER,
        is_fraud INTEGER DEFAULT 0,
        FOREIGN KEY (transaction_id) REFERENCES Transaction_record(transaction_id),
        FOREIGN KEY (rule_id) REFERENCES Fraud_rule(rule_id)
    );
    """)
    conn.commit()
    conn.close()

def populate_initial_data(db_path=DB_PATH):
    """Populates basic master data and generates realistic transactions for ML model training."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Master Data Check
    cursor.execute("SELECT COUNT(*) FROM Customer")
    if cursor.fetchone()[0] > 0:
        conn.close()
        return

    # 1. Customers
    customers = []
    first_names = ['Rahul', 'Priya', 'Amit', 'Sneha', 'Vikram', 'Anita', 'Suresh', 'Meena', 'Rohan', 'Pooja',
                   'Karan', 'Neha', 'Deepak', 'Kavita', 'Manish', 'Swati', 'Arjun', 'Divya', 'Sanjay', 'Ritu']
    last_names = ['Sharma', 'Patel', 'Kumar', 'Gupta', 'Singh', 'Rao', 'Reddy', 'Joshi', 'Mehta', 'Verma',
                  'Kapoor', 'Nair', 'Malhotra', 'Pillai', 'Agarwal', 'Deshmukh', 'Choudhury', 'Saxena', 'Menon', 'Sharma']
    cities = ['Mumbai', 'Delhi', 'Bengaluru', 'Chennai', 'Kolkata', 'Hyderabad', 'Pune', 'Ahmedabad', 'Jaipur', 'Lucknow',
              'Chandigarh', 'Kochi', 'Indore', 'Surat', 'Nagpur', 'Visakhapatnam', 'Bhopal', 'Coimbatore', 'Patna', 'Vadodara']

    for i in range(1, 21):
        customers.append((
            i, first_names[i-1], last_names[i-1],
            f"{first_names[i-1].lower()}.{last_names[i-1].lower()}@example.com",
            f"+91-98{i:02d}-12345", "1990-05-15",
            f"{i*10} Main St, {cities[i-1]}", "Active"
        ))
    cursor.executemany("INSERT INTO Customer VALUES (?,?,?,?,?,?,?,?)", customers)

    # 2. Branches
    branches = []
    ifsc_codes = ['SBIN0001001', 'SBIN0001002', 'HDFC0002001', 'HDFC0002002', 'ICIC0003001',
                  'ICIC0003002', 'UTIB0004001', 'UTIB0004002', 'PUNB0005001', 'PUNB0005002',
                  'BARB0006001', 'BARB0006002', 'CNRB0007001', 'CNRB0007002', 'KKBK0008001',
                  'KKBK0008002', 'INDB0009001', 'INDB0009002', 'IDIB0010001', 'IDIB0010002']
    states = ['Maharashtra', 'Delhi', 'Karnataka', 'Tamil Nadu', 'West Bengal', 'Telangana', 'Maharashtra', 'Gujarat',
              'Rajasthan', 'Uttar Pradesh', 'Punjab', 'Kerala', 'Madhya Pradesh', 'Gujarat', 'Maharashtra', 'Andhra Pradesh',
              'Madhya Pradesh', 'Tamil Nadu', 'Bihar', 'Gujarat']
    
    for i in range(1, 21):
        branches.append((i, f"{cities[i-1]} Branch", cities[i-1], states[i-1], ifsc_codes[i-1]))
    cursor.executemany("INSERT INTO Branch VALUES (?,?,?,?,?)", branches)

    # 3. Accounts Details
    accounts = []
    acc_types = ['Savings', 'Checking', 'Fixed Deposit', 'Salary', 'Savings', 'Recurring Deposit']
    for i in range(1, 21):
        balance = round(float(np.random.uniform(5000, 500000)), 2)
        accounts.append((i, acc_types[i % len(acc_types)], f"ACC10020260{i:02d}", balance, '2021-01-15', 'Active', i, i))
    cursor.executemany("INSERT INTO accounts_details VALUES (?,?,?,?,?,?,?,?)", accounts)

    # 4. Transaction Types
    tx_types = [
        (1, 'Deposit', 'Direct cash or electronic deposit.'),
        (2, 'Withdrawal', 'Cash withdrawal from branch counter or ATM.'),
        (3, 'NEFT Transfer', 'National Electronic Funds Transfer.'),
        (4, 'RTGS Transfer', 'Real Time Gross Settlement high value transfer.'),
        (5, 'IMPS Transfer', 'Immediate Payment Service 24/7 instant transfer.'),
        (6, 'UPI Payment', 'Unified Payments Interface mobile payment.'),
        (7, 'ATM Cash Withdrawal', 'Debit card cash withdrawal.'),
        (8, 'Bill Payment', 'Utility bill and merchant payments.'),
        (9, 'Loan Disbursement', 'Loan credit.'),
        (10, 'Loan EMI Payment', 'Loan installment deduction.'),
        (11, 'Credit Card Payment', 'Credit card bill payment.'),
        (12, 'Interest Credit', 'Periodic interest credit.'),
        (13, 'Service Charge', 'Bank service fee deduction.'),
        (14, 'Cheque Deposit', 'Cheque credit.'),
        (15, 'Cheque Bounce Fee', 'Cheque bounce penalty.'),
        (16, 'POS Purchase', 'Merchant terminal card payment.'),
        (17, 'Online Purchase', 'E-commerce merchant payment.'),
        (18, 'FD Creation', 'Transfer to Fixed Deposit.'),
        (19, 'RD Deposit', 'Recurring deposit transfer.'),
        (20, 'Refund Credit', 'Merchant refund credit.')
    ]
    cursor.executemany("INSERT INTO Transaction_type VALUES (?,?,?)", tx_types)

    # 5. Beneficiaries
    beneficiaries = []
    for i in range(1, 21):
        beneficiaries.append((i, f"Ben_{first_names[i-1]}", f"BEN9900110{i:02d}", "HDFC Bank", ifsc_codes[i-1], '2022-01-10', i))
    cursor.executemany("INSERT INTO Beneficiary VALUES (?,?,?,?,?,?,?)", beneficiaries)

    # 6. Devices
    devices = []
    os_list = ['iOS 17.4', 'Android 14', 'Windows 11 Pro', 'macOS Sonoma', 'iPadOS 17.2']
    dev_types = ['Smartphone', 'Laptop', 'Tablet', 'Desktop']
    for i in range(1, 21):
        devices.append((i, dev_types[i % len(dev_types)], f"Model-{i}", os_list[i % len(os_list)], f"DEV880010{i:02d}", '2023-01-15', i))
    cursor.executemany("INSERT INTO Device VALUES (?,?,?,?,?,?,?)", devices)

    # 7. Fraud Rules
    rules = [
        (1, 'High Velocity Txn', 'More than 5 txns in 10 mins', 'High', 5.0, 'Active'),
        (2, 'Large Txn Limit', 'Single txn exceeding threshold', 'Critical', 200000.0, 'Active'),
        (3, 'Failed Login Attempt', 'Consecutive failed login attempts', 'Medium', 3.0, 'Active'),
        (4, 'New Device Login', 'Login from unrecognized device', 'Medium', 1.0, 'Active'),
        (5, 'Off-Hour Access', 'High-value transaction late night (1 AM - 4 AM)', 'High', 100000.0, 'Active'),
        (6, 'Unusual Location / Geo Mismatch', 'Geographic distance travel speed > 800 km/h', 'High', 800.0, 'Active')
    ]
    cursor.executemany("INSERT INTO Fraud_rule VALUES (?,?,?,?,?,?)", rules)

    conn.commit()

    # 8. Generate 3,000 Synthetic Transactions with Embedded Fraud Scenarios for Data Science Training
    print("Generating synthetic dataset (3,000 records)...")
    np.random.seed(42)

    city_coords = {
        'Mumbai': (19.07609, 72.87742),
        'Delhi': (28.61393, 77.20902),
        'Bengaluru': (12.97159, 77.59456),
        'Chennai': (13.08268, 80.27071),
        'Kolkata': (22.57264, 88.36389),
        'Hyderabad': (17.38504, 78.48667),
        'Pune': (18.52043, 73.85674),
        'Ahmedabad': (23.02250, 72.57136),
        'Jaipur': (26.91243, 75.78727),
        'Lucknow': (26.84670, 80.94615)
    }

    start_date = datetime(2024, 1, 1, 0, 0, 0)
    tx_list = []
    loc_list = []
    login_list = []
    suspicious_list = []

    tx_id = 1
    susp_id = 1
    login_id = 1

    for cust_id in range(1, 21):
        cust_home_city = cities[(cust_id - 1) % len(cities)]
        home_lat, home_lon = city_coords.get(cust_home_city, (19.076, 72.877))

        # Generate ~150 transactions per customer over 6 months
        curr_time = start_date + timedelta(hours=np.random.randint(0, 48))
        
        for _ in range(150):
            curr_time += timedelta(minutes=int(np.random.exponential(scale=120)))
            
            # Determine if this transaction will be fraudulent (~5% base fraud rate + specific patterns)
            is_fraud_scenario = np.random.rand() < 0.05
            scenario_type = None

            if is_fraud_scenario:
                scenario_type = np.random.choice(['velocity', 'large_amount', 'impossible_travel', 'off_hours'])

            # Default parameters
            amount = float(np.random.lognormal(mean=7.5, sigma=1.2)) # ~1,800 INR median
            amount = round(min(max(amount, 100.0), 450000.0), 2)
            tx_status = 'Completed'
            tx_type_id = np.random.choice([3, 4, 5, 6, 7, 16, 17])
            city = cust_home_city
            lat = home_lat + np.random.uniform(-0.05, 0.05)
            lon = home_lon + np.random.uniform(-0.05, 0.05)
            dev_id = cust_id
            login_status = 'Success'
            rule_id = None
            risk_score = np.random.randint(10, 45)

            if scenario_type == 'velocity':
                # Rapid burst of transactions within 5 minutes
                amount = round(float(np.random.uniform(15000, 80000)), 2)
                curr_time += timedelta(seconds=np.random.randint(10, 120))
                rule_id = 1
                risk_score = np.random.randint(80, 98)
                
            elif scenario_type == 'large_amount':
                amount = round(float(np.random.uniform(250000, 490000)), 2)
                rule_id = 2
                risk_score = np.random.randint(85, 99)

            elif scenario_type == 'impossible_travel':
                # Jump to distant city (e.g., Delhi to Chennai in 10 mins)
                distant_city = 'Chennai' if cust_home_city != 'Chennai' else 'Delhi'
                city = distant_city
                lat, lon = city_coords[distant_city]
                rule_id = 6
                risk_score = np.random.randint(82, 97)

            elif scenario_type == 'off_hours':
                # Late night transaction between 1 AM and 4 AM
                curr_time = curr_time.replace(hour=np.random.choice([1, 2, 3]))
                amount = round(float(np.random.uniform(90000, 220000)), 2)
                rule_id = 5
                risk_score = np.random.randint(75, 95)

            # Insert Login
            login_list.append((
                login_id, f"103.{cust_id}.{np.random.randint(1,250)}.{np.random.randint(1,250)}",
                curr_time.strftime('%Y-%m-%d %H:%M:%S'), city, lat, lon, login_status, cust_id, dev_id
            ))

            # Insert Transaction
            ref_num = f"TXN{curr_time.strftime('%Y%m%d%H%M%S')}{tx_id:04d}"
            tx_list.append((
                tx_id, amount, curr_time.strftime('%Y-%m-%d %H:%M:%S'), tx_status,
                ref_num, f"Txn via type {tx_type_id}", cust_id, cust_id, tx_type_id
            ))

            # Insert Location
            loc_list.append((tx_id, city, 'State', 'India', lat, lon, tx_id))

            # Insert Suspicious Record if Flagged
            is_fraud_val = 1 if is_fraud_scenario else 0
            if is_fraud_val == 1 or risk_score > 70:
                suspicious_list.append((
                    susp_id, risk_score, curr_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'Confirmed' if is_fraud_val == 1 else 'Cleared', tx_id, rule_id or 1, is_fraud_val
                ))
                susp_id += 1

            tx_id += 1
            login_id += 1

    cursor.executemany("INSERT INTO Login_activitys VALUES (?,?,?,?,?,?,?,?,?)", login_list)
    cursor.executemany("INSERT INTO Transaction_record VALUES (?,?,?,?,?,?,?,?,?)", tx_list)
    cursor.executemany("INSERT INTO Transaction_location VALUES (?,?,?,?,?,?,?)", loc_list)
    cursor.executemany("INSERT INTO Suspicious_transaction VALUES (?,?,?,?,?,?,?)", suspicious_list)

    conn.commit()
    conn.close()
    print(f"Data insertion complete! Generated {len(tx_list)} transactions.")

if __name__ == "__main__":
    init_db()
    populate_initial_data()
