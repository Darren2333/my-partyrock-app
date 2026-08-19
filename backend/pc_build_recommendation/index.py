import json
import boto3
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize Bedrock client
bedrock_runtime = boto3.client('bedrock-runtime', region_name='ap-southeast-1')

# Bedrock model ID
MODEL_ID = 'global.anthropic.claude-haiku-4-5-20251001-v1:0-20260217-v1:0'


def build_prompt(budget, currency, use_case):
    """Build the prompt for Bedrock"""
    return f"""You are an expert PC builder with deep knowledge of hardware compatibility, price-to-performance ratios, and component availability. A user wants to build a PC with a budget of {budget} {currency} for the following primary use case: {use_case}.

Generate a complete, detailed PC build recommendation tailored to their budget and use case. Structure your response as follows:

**Recommended PC Build**

Use Case: {use_case} | Budget: {budget} {currency}

For each component listed below, provide the specific model name, approximate price in {currency}, and a brief explanation of why it was chosen for this use case and budget:

1. CPU - Include model, approximate price, and reasoning

2. GPU - Include model, approximate price, and reasoning (if no dedicated GPU is needed, explain why)

3. Motherboard - Include model, approximate price, and compatibility notes

4. RAM - Include capacity, speed, model, and price

5. Storage - Include type (SSD/HDD/NVMe), capacity, model, and price

6. Power Supply - Include wattage, efficiency rating, model, and price

7. Case - Include model, form factor, and price

8. Cooling Solution - Include type (air/liquid), model, and price

**Total Estimated Cost:** Sum up all component prices in {currency} and compare to the budget. Note if there is headroom or if compromises were made.

**Why This Build Works for {use_case}:** Write 2-3 sentences explaining how this build is optimized for the stated use case.

**Compatibility Notes and Building Tips:** List any important compatibility considerations, BIOS update requirements, installation tips, or upgrade paths to keep in mind.

Be specific with model names and realistic with current market pricing in the specified currency. Adjust component recommendations based on regional availability and pricing differences for the selected currency. If the budget is tight, prioritize the components that matter most for the use case."""


def generate_response(budget, currency, use_case):
    """Generate PC build recommendation from Bedrock"""
    prompt = build_prompt(budget, currency, use_case)

    # Build the messages for Claude
    messages = [
        {
            'role': 'user',
            'content': prompt
        }
    ]

    # Invoke model
    try:
        response = bedrock_runtime.invoke_model(
            modelId=MODEL_ID,
            messages=messages,
            system="You are a helpful PC hardware expert providing detailed build recommendations."
        )

        # Parse response
        response_body = json.loads(response['body'].read())
        return response_body['content'][0]['text']
    except Exception as e:
        logger.error(f"Bedrock error: {str(e)}")
        return f"Error generating recommendation: {str(e)}"


def lambda_handler(event, context):
    """Main Lambda handler"""
    logger.info(f"Received event: {json.dumps(event)}")
    
    # Handle CORS preflight
    if event.get('requestContext', {}).get('http', {}).get('method') == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'POST,OPTIONS',
            },
            'body': ''
        }
    
    try:
        # Parse request body
        body = json.loads(event.get('body', '{}'))
        
        budget = body.get('budget')
        currency = body.get('currency', 'USD')
        use_case = body.get('use_case', 'General Use')
        
        if not budget:
            return {
                'statusCode': 400,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Content-Type': 'application/json',
                },
                'body': json.dumps({'error': 'Missing budget parameter'})
            }
        
        # Validate inputs
        try:
            budget = int(budget)
            if budget < 500 or budget > 100000:
                budget = 2500
        except (ValueError, TypeError):
            budget = 2500
        
        # Generate recommendation
        recommendation = generate_response(budget, currency, use_case)
        
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'text/plain; charset=utf-8',
            },
            'body': recommendation
        }
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json',
            },
            'body': json.dumps({'error': f'Internal Server Error: {str(e)}'})
        }
