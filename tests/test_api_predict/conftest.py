from datetime import datetime

def pytest_report_header(config):
    """Custom header entry for pytest report"""
    timestr = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return f"Session start: {timestr}"
