import os
import importlib
import flask
from flask.cli import FlaskGroup

from full_pipeline_server import app

# @app.route("/",methods=["GET"])
# def parse_get():
#     global p
#     txt=flask.request.args.get("text")
#     if not txt:
#         return "You need to specify ?text=sometext",400
#     res=p.parse(txt)
#     return flask.Response(res,mimetype="text/plain; charset=utf-8")

# @app.route("/text/fi/parse",methods=["POST"])
# def parse_post():
#     global p,max_char
#     txt=flask.request.get_data(as_text=True)
#     if max_char>0:
#         txt=txt[:max_char]
#     if not txt:
#         return """You need to post your data as a single string. An example request would be curl --request POST --data 'Tämä on testilause' http://localhost:7689\n\n\n""",400
#     else:
#         res=p.parse(txt)
#     return flask.Response(res,mimetype="text/plain; charset=utf-8")

app.add_url_rule("/text/fi/parse", view_func = parse_post, methods=["POST"])

cli = FlaskGroup(app)

if __name__ == '__main__':
    cli()
