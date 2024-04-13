import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from sshtunnel import SSHTunnelForwarder

load_dotenv()

server = SSHTunnelForwarder(
    (os.environ.get("V13_FORK_HOST"), os.environ.get("V13_FORK_PORT")),
    ssh_username=os.environ.get("V13_FORK_SSH_USERNAME"),
    ssh_private_key=os.environ.get("SSH_PRIVATE_KEY_PATH"),
    remote_bind_address=(
        os.environ.get("V13_FORK_DB_HOST"),
        os.environ.get("V13_FORK_DB_PORT"),
    ),
)

try:
    server.start()
    print("Server connected")
except:
    print("Server not connected")

template = """Based on the table schema below, write a SQL query that would answer the user's question, given a Python dictionary called `filters` wherein the keys are the filters specified below:
{schema}

Question: {question}
SQL Query:"""

db_uri = f"mysql+pymysql://{os.environ.get('V13_FORK_DB_USERNAME')}:{os.environ.get('V13_FORK_DB_PASSWORD')}@{os.environ.get('V13_FORK_DB_HOST')}:{server.local_bind_port}/{os.environ.get('V13_FORK_DB_USERNAME')}"
db = SQLDatabase.from_uri(
    db_uri,
    include_tables=[
        "tabPurchase Order",
        "tabPurchase Order Item",
        "tabPurchase Receipt",
        "tabPurchase Receipt Item",
        "tabPurchase Invoice",
        "tabPurchase Invoice Item",
        "tabBatch",
    ],
    sample_rows_in_table_info=0,
)

print("Connected to database")

# print(db.run("select * from `tabSales Invoice` limit 1"))

# print(db.get_table_info(table_names=["tabSales Invoice"]))


def get_schema(_):
    return db.get_table_info()


def run_query(query):
    return db.run(query)


prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI(
    openai_api_key=os.environ.get("OPENAI_API_KEY"), model="gpt-4-turbo-preview"
)

sql_response = (
    RunnablePassthrough.assign(schema=get_schema)
    | prompt
    | model.bind(stop=["\nSQLResult:"])
    | StrOutputParser()
)

print(
    sql_response.invoke(
        {
            "question": """
    Business Question:
        - What are the received purchased items on a certain period? Are there over/under received purchased items?
        - What are the unreceived ordered items?
                           
    Description: An item-level report that shows received purchase orders and purchase orders expected to be received.
                           
    Filters and Parameters:
        - Company: Reference only transactions and branches of the selected company
        - Purchase Receipt Start Date
            - If received, based on the Purchase Receipt date. If still unreceived, based on the Required By Date defined in the Purchase Order.
            - Include only purchase orders / purchase receipts during or after this date
        - Purchase Receipt End Date
            - If received, based on the Purchase Receipt date. If still unreceived, based on the Required By Date defined in the Purchase Order.
            -Include only purchase orders / purchase receipts before or during this date
        - Warehouse: Include transactions in the selected warehouse
        - Supplier: Include only items sold by the selected supplier
        - Brand (multi-select)
            - Include only items of the Brand/s selected
        - Item: Include only item selected
        - Purchase Order #: Show only items in the specified PO #
        - Supplier’s Sales Invoice: Show only items in the specified Sales Invoice (defined in the Purchase Receipt)
        - Purchase Receipt #: Show only items in the specified PR #
        - Status (radio button):
            - All – include both unreceived PO items and received items through the Purchase Receipt (Default Value)
                - Unreceived items from Purchase Orders with Closed status should be excluded
            - Received – include only ordered items that are received already through the Purchase Receipt
            - Unreceived – include only ordered items that are still waiting to be received
                - Unreceived items from Purchase Orders with Closed status should be excluded
        - Over/Under Received Only (checkbox):
            - If checked, include only received items where the quantity received does not match the quantity ordered.
            - Defaults to unchecked

    Fields:
        - Purchase Order #: Show the Purchase Order No. (name)
        - Purchase Order Date: Show the Purchase Order Date
        - Expected Delivery Date: Show the Required By Date in the Purchase Order
        - Supplier: Show the Supplier in the Purchase Order
        - Supplier's Sales Invoice: Show the Supplier Invoice No. in the Purchase Invoice (if any)
        - Purchase Receipt #: Show the Purchase Receipt No. (name)
        - Purchase Receipt Date: Show the Purchase Receipt Date
        - Receiving Warehouse: Show the warehouse in the Purchase Receipt. If not yet received, show the warehouse in the Purchase Order.
        - Barcode: Show the barcode in the Purchase Receipt. If not yet received, show the barcode in the Purchase Order.
        - Item Code: Item Name: Show the Item Code: Item Name in the Purchase Order
        - Batch: Batch No in the Purchase Receipt
        - Expiration Date: Show the Expiration Date of the received Batch
        - Purchase Order UOM: UOM in the Purchase Order
        - Expected Qty: Ordered Quantity in the Purchase Order
        - Purchase Receipt UOM: UOM in the Purchase Receipt
        - Received Qty: Received Quantity in the Purchase Receipt
        - Over/Under Received Qty:
            - If the item is partially or fully received, Received Qty - Expected Qty
            - If the item is still unreceived, it should be empty
        - Receiver: Show user who created the Purchase Receipt

    Notes:
        - Prioritize joins between item/child tables
"""
        }
    )
)

try:
    server.stop()
    print("Server disconnected")
except:
    print("Server not disconnected")
