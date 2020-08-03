import logging

import azure.functions as func

def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
        from .predict import predict_image_from_url
        image_url = req.params.get('img')
        if image_url:
            logging.info(image_url)

            results = predict_image_from_url(image_url)

            headers = {
                "Content-type": "application/json",
                "Access-Control-Allow-Origin": "*"
            }

            return func.HttpResponse(json.dumps(results), headers = headers)
        else:
            return func.HttpResponse(
                    "hello",
                    status_code=200
                )
    
    except Exception as e:
        logging.info("Exception: "+str(e))
        return func.HttpResponse(
                    "Exception: "+str(e),
                    status_code=200
                )