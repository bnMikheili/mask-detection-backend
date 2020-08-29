import asyncio
import websockets
import json
from server.ConnectionContainer import ConnectionContainer

# Create container
connection_container = ConnectionContainer()

# WebSocket server
async def server(websocket, path):
    # Take request parse and send message back.
    async for message in websocket:
        params = json.loads(message)

        # Taking video streaminga sdp
        answer = await connection_container.handle_offer(sdp=params["sdp"])
        await websocket.send(answer.sdp)


# Start websocket server
start_server = websockets.serve(server, "localhost", 5000)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
