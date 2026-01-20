import os
from datetime import datetime
import requests


# definition of the API address and port
API_ADDRESS = os.environ.get('API_ADDRESS', default='127.0.0.1')
API_PORT = os.environ.get('API_PORT', default='8000')


def log(logString):
    # print result
    if os.environ.get('LOG_STDOUT') == '1':
        print(logString)

    # log to file
    if os.environ.get('LOG_FILE') == '1':
        logDir = os.environ.get('LOGDIR', default='.')
        logFn = os.path.join(logDir, 'test_predict_api.log')
        with open(logFn, 'a') as file:
            file.write(logString)
            file.write('\n\n')


def test_health():
    """Test the health check API endpoint"""

    expected_status_code = 200
    expected_status_str = 'healthy'

    outputStr = '\n'.join((
        '=========================',
        'Health test',
        'TIME: {time}',
        '=========================',
        'request done at "/health"',
        'expected status code = {expected_status_code}',
        'actual status code = {status_code}',
        'expected status string = {expected_status_str}',
        'actual status string = {status_str}',
        '==>  {test_status}'
    ))

    url=f'http://{API_ADDRESS}:{API_PORT}/health'
    r = requests.get(url=url)

    # query status and content
    status_code = r.status_code
    content = r.json()

    # prepare test result string
    if ((status_code == expected_status_code) and
        (content['status'] == expected_status_str)
    ):
        test_status = 'SUCCESS'
    else:
        test_status = 'FAILURE'

    resultStr = outputStr.format(
        time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        expected_status_code=expected_status_code,
        status_code=status_code, 
        expected_status_str=expected_status_str,
        status_str=content['status'], 
        test_status=test_status
    )

    log(resultStr)


def test_info():
    """Test the model info API endpoint"""

    expected_status_code = 200
    expected_content_types = {
        "model_path": str,
        "num_classes": int,
        "device": str
    }

    outputStr = '\n'.join((
        '=========================',
        'Model Info Test',
        'TIME: {time}',
        '=========================',
        'request done at "/model/info"',
        'expected status = {expected_status_code}',
        'actual status = {status_code}',
        'expected content types = {expected_content_types}',
        'actual content types = {content_types}',
        '==>  {test_status}'
    ))

    url=f'http://{API_ADDRESS}:{API_PORT}/model/info'
    r = requests.get(url=url)

    # query status and content
    status_code = r.status_code
    content = r.json()
    content_types={k: type(v) for k,v in content.items()}

    # prepare test result string
    if ((status_code == expected_status_code) and
        (content_types == expected_content_types)
    ):
        test_status = 'SUCCESS'
    else:
        test_status = 'FAILURE'

    resultStr = outputStr.format(
        time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        expected_status_code=expected_status_code,
        status_code=status_code,
        expected_content_types=expected_content_types,
        content_types=content_types,
        test_status=test_status
    )

    log(resultStr)


if __name__ == "__main__":
    test_health()
    test_info()
