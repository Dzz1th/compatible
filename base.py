from langchain_openai import ChatOpenAI
from typing import Optional
from pydantic import BaseModel, Field
from tqdm import tqdm

import pandas as pd

CUSTOM_PROMPT = """
    You are a trader in a large investment bank that specialize on FX derivatives. Your task is to decompose the description of the trade into the contract and all the details that are required to understand it.
    The details should include: Instrument Type, Reference Rates, Details for each leg (Rates, Frequency, Day Count Convention), Reset Frequency for floating legs, Spreads, Notionals, Tenor, Effective Date and Maturity Data.
    Reference rates should be stated exactly as they are in the text, do not guess them if they are not provided directly. For example 3m Term SOFR should not be stated as SOFR.
    Remember that you consider trades from yours perspective (as a bank). So you should provide receiving and paying legs for you as a bank. It means that if the client pays, you receive, and backwards. If bank pays - then you pay, and if bank receive - you receive. If you buy swap - you receive floating leg and pay fixed, if sell - you receive fixed leg and pay floating. So you should provide floating leg and fixed leg for YOU.
    For Cross-Currency Swaps set notional only to the legs for which it is directly provided, $ sign doesn't count for direct provision. For Floating-Fixed Swaps in 1 currency assume that legs notionals are equal.
    If some details are not provided in the text - say that they are not provided.
    Please think before giving an answer.

    ##Trade description:
    {trade_description}
"""

STRUCTURING_PROMPT = """
    You are a trader in a large investment bank that specialize on FX derivatives.
    Your task is to transform the description of the FX Swap contract into the structured JSON object. 
    You are on the side of the bank, so you should consider paying and receiving leg from your perspective.
    For reference indexes please include the entire name, f.e. 6M SOFT or 3M EURIBOR, do not truncate that.
    
    Here are the fields that you should include:
    ['EffectiveDate', 'MaturityDate', 'TenorYears', 'PayLegNotional', 'PayLegCcy', 'PayLegFreqMonths', 'PayLegBasis', 'PayLegFloatIndex', 'PayLegFloatSpreadBp', 'PayLegFixedRatePct', 'RecLegNotional', 'RecLegCcy', 'RecLegFreqMonths', 'RecLegBasis', 'RecLegFloatIndex', 'RecLegFloatSpreadBp', 'RecLegFixedRatePct']

    If there are no information provided on the certain field - set it to None.

    ##Trade description:
    {trade_description}
"""


test_data = [
    {
        'trade_description': 'Sell 10y SOFR swap at 3.45%',
        'ground_truth': {
            'EffectiveDate': None,
            'MaturityDate': None,
            'TenorYears': 10,
            'PayLegNotional': None,
            'PayLegCcy': None,
            'PayLegFreqMonths': None,
            'PayLegBasis': None,
            'PayLegFloatIndex': 'SOFR',
            'PayLegFloatSpreadBp': None,
            'PayLegFixedRatePct': None,
            'RecLegNotional': None,
            'RecLegCcy': None,
            'RecLegFreqMonths': None,
            'RecLegBasis': None,
            'RecLegFloatIndex': None,
            'RecLegFloatSpreadBp': None, #0 -> None
            'RecLegFixedRatePct': 3.45
        }
    },
    {
        'trade_description': "USDEUR cross currency swap (fixed/fixed), 10 y maturity Client pays 2.3% USD, we pay 1.7% EUR",
        "ground_truth": {'EffectiveDate': None,
                'MaturityDate': None,
                'TenorYears': 10,
                'PayLegNotional': None,
                'PayLegCcy': 'EUR',
                'PayLegFreqMonths': None,
                'PayLegBasis': None,
                'PayLegFloatIndex': None,
                'PayLegFloatSpreadBp': None,
                'PayLegFixedRatePct': 1.7,
                'RecLegNotional': None,
                'RecLegCcy': 'USD',
                'RecLegFreqMonths': None,
                'RecLegBasis': None,
                'RecLegFloatIndex': None,
                'RecLegFloatSpreadBp': None,
                'RecLegFixedRatePct': 2.3}
    },
    {
        'trade_description': "Vanilla swap at 2.90%, we pay fixed, 10y maturity",
        'ground_truth': {'EffectiveDate': None,
            'MaturityDate': None,
            'TenorYears': 10,
            'PayLegNotional': None,
            'PayLegCcy': None,
            'PayLegFreqMonths': None,
            'PayLegBasis': None,
            'PayLegFloatIndex': None,
            'PayLegFloatSpreadBp': None,
            'PayLegFixedRatePct': 2.90,
            'RecLegNotional': None,
            'RecLegCcy': None,
            'RecLegFreqMonths': None,
            'RecLegBasis': None,
            'RecLegFloatIndex': None,
            'RecLegFloatSpreadBp': None,
            'RecLegFixedRatePct': None}
    },
    {
        'trade_description': "Notional: 10m Bank pays: 6M Term SOFR, semi-annual, act/360 Bank receives: 3.45%, semi-annual, 30/360 Start date: 10 November 2009 Tenor: 5y",
        'ground_truth': {'EffectiveDate': '2009-11-10',
            'MaturityDate': None,
            'TenorYears': 5,
            'PayLegNotional': 10000000,
            'PayLegCcy': None,
            'PayLegFreqMonths': 6,
            'PayLegBasis': 'Act/360',
            'PayLegFloatIndex': '6M Term SOFR',
            'PayLegFloatSpreadBp': None,
            'PayLegFixedRatePct': None,
            'RecLegNotional': 10000000,
            'RecLegCcy': None,
            'RecLegFreqMonths': 6,
            'RecLegBasis': '30/360',
            'RecLegFloatIndex': None,
            'RecLegFloatSpreadBp': None,
            'RecLegFixedRatePct': 3.45}
    },
    {
        'trade_description': 'Sell SOFR swap at three point five percent, maturity November 18, 2027',
        'ground_truth': {'EffectiveDate': None,
                'MaturityDate': '2027-11-18',
                'TenorYears': None,
                'PayLegNotional': None,
                'PayLegCcy': None,
                'PayLegFreqMonths': None,
                'PayLegBasis': None,
                'PayLegFloatIndex': 'SOFR',
                'PayLegFloatSpreadBp': None,
                'PayLegFixedRatePct': None,
                'RecLegNotional': None,
                'RecLegCcy': None,
                'RecLegFreqMonths': None,
                'RecLegBasis': None,
                'RecLegFloatIndex': None,
                'RecLegFloatSpreadBp': None,
                'RecLegFixedRatePct': 3.5}
    },
    {
        'trade_description': 'Paying USD, receiving JPY, 2-year tenor, plus 32bps spread, ¥5bn notional',
        'ground_truth': {'EffectiveDate': None,
                'MaturityDate': None,
                'TenorYears': 2,
                'PayLegNotional': None,
                'PayLegCcy': 'USD',
                'PayLegFreqMonths': None,
                'PayLegBasis': None,
                'PayLegFloatIndex': None,
                'PayLegFloatSpreadBp': None,
                'PayLegFixedRatePct': None,
                'RecLegNotional': 5000000000,
                'RecLegCcy': 'JPY',
                'RecLegFreqMonths': None,
                'RecLegBasis': None,
                'RecLegFloatIndex': None,
                'RecLegFloatSpreadBp': 32,
                'RecLegFixedRatePct': None}
    },
    {
        'trade_description': 'Executed 2-year basis swap Paying EUR, receiving USD plus 85 basis points 100mm USD notional',
        'ground_truth': {'EffectiveDate': None,
            'MaturityDate': None,
            'TenorYears': 2,
            'PayLegNotional': None,
            'PayLegCcy': 'EUR',
            'PayLegFreqMonths': None,
            'PayLegBasis': None,
            'PayLegFloatIndex': None,
            'PayLegFloatSpreadBp': None,
            'PayLegFixedRatePct': None,
            'RecLegNotional': 100000000,
            'RecLegCcy': 'USD',
            'RecLegFreqMonths': None,
            'RecLegBasis': None,
            'RecLegFloatIndex': None,
            'RecLegFloatSpreadBp': 85,
            'RecLegFixedRatePct': None}
    },
    {
        'trade_description': 'I have trade details Profile: USD Interest rate swap Tenor: 10 years, both legs quarterly We pay USD fixed 3.35%, client pays SOFR',
        'ground_truth': {'EffectiveDate': None,
                'MaturityDate': None,
                'TenorYears': 10,
                'PayLegNotional': None,
                'PayLegCcy': 'USD',
                'PayLegFreqMonths': 3,
                'PayLegBasis': None,
                'PayLegFloatIndex': None,
                'PayLegFloatSpreadBp': None,
                'PayLegFixedRatePct': 3.35,
                'RecLegNotional': None,
                'RecLegCcy': 'USD',
                'RecLegFreqMonths': 3,
                'RecLegBasis': None,
                'RecLegFloatIndex': 'SOFR',
                'RecLegFloatSpreadBp': None,
                'RecLegFixedRatePct': None}
    },
    {
        'trade_description': "Swapped $300M—paying 3.1% fixed, taking SOFR floating. Semi-annual. 5y from Feb '22.",
        'ground_truth': {'EffectiveDate': '2022-02-01',
                            'MaturityDate': None,
                            'TenorYears': 5,
                            'PayLegNotional': 300000000,
                            'PayLegCcy': 'USD',
                            'PayLegFreqMonths': 6,
                            'PayLegBasis': None,
                            'PayLegFloatIndex': None,
                            'PayLegFloatSpreadBp': None,
                            'PayLegFixedRatePct': 3.1,
                            'RecLegNotional': 300000000,
                            'RecLegCcy': 'USD',
                            'RecLegFreqMonths': 6,
                            'RecLegBasis': None,
                            'RecLegFloatIndex': 'SOFR',
                            'RecLegFloatSpreadBp': None,
                            'RecLegFixedRatePct': None
                        }
    },
    {
        'trade_description': "Done $150M fixed payer. 3.5% fixed vs. SOFR float. Semi-annual resets. Runs 7y.",
        'ground_truth': {'EffectiveDate': None,
                            'MaturityDate': None,
                            'TenorYears': 7,
                            'PayLegNotional': 150000000,
                            'PayLegCcy': 'USD',
                            'PayLegFreqMonths': None,
                            'PayLegBasis': None,
                            'PayLegFloatIndex': None,
                            'PayLegFloatSpreadBp': None,
                            'PayLegFixedRatePct': 3.5,
                            'RecLegNotional': 150000000,
                            'RecLegCcy': 'USD',
                            'RecLegFreqMonths': 6,
                            'RecLegBasis': None,
                            'RecLegFloatIndex': 'SOFR',
                            'RecLegFloatSpreadBp': None,
                            'RecLegFixedRatePct': None}
    },
    {
        'trade_description': "$100M vanilla swap—fixed payer at 4.1%, 6M Term SOFR floater. Semi-annual resets. 7y, starts Jan '24.",
        'ground_truth': {'EffectiveDate': '2024-01-01',
                        'MaturityDate': None,
                        'TenorYears': 7,
                        'PayLegNotional': '100000000',
                        'PayLegCcy': 'USD',
                        'PayLegFreqMonths': 6,
                        'PayLegBasis': None,
                        'PayLegFloatIndex': None,
                        'PayLegFloatSpreadBp': None,
                        'PayLegFixedRatePct': 4.1,
                        'RecLegNotional': 100000000,
                        'RecLegCcy': 'USD',
                        'RecLegFreqMonths': 6,
                        'RecLegBasis': None,
                        'RecLegFloatIndex': '6M Term SOFR',
                        'RecLegFloatSpreadBp': None,
                        'RecLegFixedRatePct': None}
    },
    {
        'trade_description': "Booked $250M basis swap. Pay 3M SOFR + 15bps, receive 6M SOFR. Rolling quarterly vs. semi. 3y.",
        'ground_truth': {'EffectiveDate': None,
                    'MaturityDate': None,
                    'TenorYears': 3,
                    'PayLegNotional': 250000000,
                    'PayLegCcy': 'USD',
                    'PayLegFreqMonths': 3,
                    'PayLegBasis': None,
                    'PayLegFloatIndex': '3M SOFR',
                    'PayLegFloatSpreadBp': 15,
                    'PayLegFixedRatePct': None,
                    'RecLegNotional': 250000000,
                    'RecLegCcy': 'USD',
                    'RecLegFreqMonths': 6,
                    'RecLegBasis': None,
                    'RecLegFloatIndex': '6M SOFR',
                    'RecLegFloatSpreadBp': None,
                    'RecLegFixedRatePct': None}
    },
    {
        'trade_description': "$75M fixed-for-float. Pay 3.25% USD fixed, receive 6M EURIBOR flat. Semi-annual. 7y deal.",
        'ground_truth': {'EffectiveDate': None,
                'MaturityDate': None,
                'TenorYears': 7,
                'PayLegNotional': 75000000,
                'PayLegCcy': 'USD',
                'PayLegFreqMonths': 6,
                'PayLegBasis': None,
                'PayLegFloatIndex': None,
                'PayLegFloatSpreadBp': None,
                'PayLegFixedRatePct': 3.25,
                'RecLegNotional': None,
                'RecLegCcy': 'EUR',
                'RecLegFreqMonths': 6,
                'RecLegBasis': None,
                'RecLegFloatIndex': '6M EURIBOR',
                'RecLegFloatSpreadBp': None,
                'RecLegFixedRatePct': None}
    },
    {
        'trade_description': "$200M forward-starting swap: Pay fixed 3.2%, receive SOFR. Starts 1 Jan 2025, runs 5y from then. Semi rolls.",
        'ground_truth': {'EffectiveDate': '2025-01-01',
                'MaturityDate': None,
                'TenorYears': 5,
                'PayLegNotional': 200000000,
                'PayLegCcy': 'USD',
                'PayLegFreqMonths': 6,
                'PayLegBasis': None,
                'PayLegFloatIndex': None,
                'PayLegFloatSpreadBp': None,
                'PayLegFixedRatePct': 3.2,
                'RecLegNotional': 200000000,
                'RecLegCcy': 'USD',
                'RecLegFreqMonths': 6,
                'RecLegBasis': None,
                'RecLegFloatIndex': 'SOFR',
                'RecLegFloatSpreadBp': None,
                'RecLegFixedRatePct': None}
    }
]

class Swap(BaseModel):
    EffectiveDate: Optional[str] = Field(description="The date on which the swap contract begins, and cash flow calculations start. Date in format MM/DD/YYYY.")
    MaturityDate: Optional[str] = Field(description="The date on which the swap terminates, marking the end of cash flow exchanges. Date in format MM/DD/YYYY.")
    TenorYears: Optional[str] = Field(description="The duration of the swap contract expressed in years. In case of months could be a fraction.")
    PayLegNotional: Optional[str] = Field(description="The principal amount for the pay leg, used to calculate payments. Should be translated into the actual number (10M -> 10000000)")
    PayLegCcy: Optional[str] = Field(description="The currency in which payments on the pay leg are made. 3 letter ISO code.")
    PayLegFreqMonths: Optional[str] = Field(description="The frequency of payments on the pay leg, expressed in months.")
    PayLegBasis: Optional[str] = Field(description="The day count convention used for the pay leg to calculate accrued interest. Should be in a format {Month Convention}/{Year Convention}.")
    PayLegFloatIndex: Optional[str] = Field(description="The floating rate index for the pay leg (if applicable). For example SOFR. It should be included as it is stated in the trade description. Leave None if the pay leg is fixed.")
    PayLegFloatSpreadBp: Optional[str] = Field(description="The spread added to the floating rate on the pay leg, expressed in basis points (bps). Leave None if the pay leg is fixed.")
    PayLegFixedRatePct: Optional[str] = Field(description="The fixed interest rate on the pay leg, expressed as a percentage. For example 3.45. Leave None if the paying leg is floating")
    RecLegNotional: Optional[str] = Field(description="The principal amount for the receive leg, used to calculate payments. Should be translated into the actual number (10M -> 10000000)")
    RecLegCcy: Optional[str] = Field(description="The currency in which payments on the receive leg are made. 3 letter ISO code.")
    RecLegFreqMonths: Optional[str] = Field(description="The frequency of payments on the receive leg, expressed in months.")
    RecLegBasis: Optional[str] = Field(description="The day count convention used for the receive leg to calculate accrued interest. Should be in a format {Month Convention}/{Year Convention}.")
    RecLegFloatIndex: Optional[str] = Field(description="The floating rate index for the receive leg (if applicable). For example SOFR. It should be included as it is stated in the trade description. Leave None if the receive leg is fixed.")
    RecLegFloatSpreadBp: Optional[str] = Field(description="The spread added to the floating rate on the receive leg, expressed in basis points (bps). Leave None if the receive leg is fixed.")
    RecLegFixedRatePct: Optional[str] = Field(description="The fixed interest rate on the receive leg, expressed as a percentage. For example 3.45. Leave None if the receiving leg is floating")


def predict(trade) -> Swap:
    trade_description = chat_description.invoke(CUSTOM_PROMPT.format(trade_description=trade)).content
    swap = structured_model.invoke(STRUCTURING_PROMPT.format(trade_description=trade_description))
    return swap

def parse_swap(swap: dict) -> dict:
    if swap['TenorYears'] is not None:
        swap['TenorYears'] = int(swap['TenorYears'])
    if swap['PayLegNotional'] is not None:
        swap['PayLegNotional'] = int(swap['PayLegNotional'])
    if swap['RecLegNotional'] is not None:
        swap['RecLegNotional'] = int(swap['RecLegNotional'])
    if swap['PayLegFreqMonths'] is not None:
        swap['PayLegFreqMonths'] = int(swap['PayLegFreqMonths'])
    if swap['PayLegFloatSpreadBp'] is not None:
        if swap['PayLegFloatSpreadBp'] == '0' or swap['PayLegFloatSpreadBp'] == '':
            swap['PayLegFloatSpreadBp'] = None
        else:
            swap['PayLegFloatSpreadBp'] = float(swap['PayLegFloatSpreadBp'])
    if swap['PayLegFixedRatePct'] is not None:
        swap['PayLegFixedRatePct'] = float(swap['PayLegFixedRatePct'])
    if swap['RecLegNotional'] is not None:
        swap['RecLegNotional'] = int(swap['RecLegNotional'])
    if swap['RecLegFreqMonths'] is not None:
        swap['RecLegFreqMonths'] = int(swap['RecLegFreqMonths'])
    if swap['RecLegFloatSpreadBp'] is not None:
        if swap['RecLegFloatSpreadBp'] == '0' or swap['RecLegFloatSpreadBp'] == '':
            swap['RecLegFloatSpreadBp'] = None
        else:
            swap['RecLegFloatSpreadBp'] = float(swap['RecLegFloatSpreadBp'])
    if swap['RecLegFixedRatePct'] is not None:
        swap['RecLegFixedRatePct'] = float(swap['RecLegFixedRatePct'])
    return swap

def score(predict, golden) -> dict:
    results = {}
    if set(predict.keys()) != set(golden.keys()): 
        raise KeyError("Keys should coincide in the predict and golden labels")
    for key in predict.keys():
        if predict[key] == golden[key]:
            results[key] = 1
        else:
            results[key] = 0
    
    return results

openai_api_key = "<your-key>"
chat_description = ChatOpenAI(temperature=0.5, model="gpt-4o", openai_api_key=openai_api_key)
chat_structured = ChatOpenAI(temperature=0, model='gpt-4o', openai_api_key=openai_api_key)

structured_model = chat_structured.with_structured_output(Swap)

## Testing pipeline ~2 min
# scores = {}
# for trade in tqdm(test_data):
#     trade_description = trade['trade_description']
#     swap = predict(trade)
#     print(swap)
#     swap = parse_swap(swap.dict())
#     ground_truth = trade.get('ground_truth', {})
#     scores[trade_description] = score(swap, ground_truth)

# accuracy_dict = {k: sum(v.values())/len(v) for k, v in scores.items()}
# total_score = sum([sum(values) for values in accuracy_dict.values()]) / sum([len(values) for values in accuracy_dict.values()])
def swap_to_record(swap: dict, id, text):
    return {
        'solution': 'ExpectedResults',
        'trade_group': 'Scoring',
        'trade_id': id, 
        'trial_id': 0,
        'entry_text': text, 
        'effective_date': swap['EffectiveDate'],
        'maturity_date': swap['MaturityDate'],
        'tenor_years': swap['TenorYears'],
        'pay_leg_notional': swap['PayLegNotional'],
        'pay_leg_ccy': swap['PayLegCcy'],
        'pay_leg_freq_months': swap['PayLegFreqMonths'],
        'pay_leg_basis': swap['PayLegBasis'],
        'pay_leg_float_index': swap['PayLegFloatIndex'],
        'pay_leg_float_spread_bp': swap['PayLegFloatSpreadBp'],
        'pay_leg_fixed_rate_pct': swap['PayLegFixedRatePct'],
        'rec_leg_notional': swap['RecLegNotional'],
        'rec_leg_ccy': swap['RecLegCcy'],
        'rec_leg_freq_months': swap['RecLegFreqMonths'],
        'rec_leg_basis': swap['RecLegBasis'],
        'rec_leg_float_index': swap['RecLegFloatIndex'],
        'rec_leg_float_spread_bp': swap['RecLegFloatSpreadBp'],
        'rec_leg_fixed_rate_pct': swap['RecLegFixedRatePct']
    }

## Predict
input_path = "./HackathonOutput.csv"
df = pd.read_csv(input_path)

results = []
for index, row in tqdm(df.iterrows()):
    trade_description = row['entry_text']
    swap = predict(trade_description)
    swap = parse_swap(swap.dict())
    record = swap_to_record(swap, index, trade_description)
    results.append(record)

results_df = pd.DataFrame(results)
results_df.to_csv('HackathonOutput.csv', index=False)