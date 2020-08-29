import asyncio
import os
import json
from aiohttp import web
import websockets
from server.ConnectionContainer import ConnectionContainer
import json

ROOT = os.path.dirname(__file__)
routes = web.RouteTableDef()

connection_container = ConnectionContainer()


async def on_shutdown():
    pass


@routes.get("/")
async def index(request):
    content = open(os.path.join(ROOT, "server/public/index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)


@routes.post("/offer")
async def offer(request):
    params = await request.json()
    answer = await connection_container.handle_offer(sdp=params["sdp"])
    return web.Response(
        content_type="application/json",
        body=json.dumps({"sdp": answer.sdp, "type": "answer"}),
    )


if __name__ == "__main__":
    app = web.Application()
    app.on_shutdown.append(on_shutdown)

    app.add_routes(routes)
    app.add_routes([web.static("/", ROOT + "server/public")])
    web.run_app(app, port=3002)
