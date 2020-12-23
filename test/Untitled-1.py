import imgkit
config = imgkit.config(wkhtmltoimage='C:/Program Files/wkhtmltopdf/bin/wkhtmltoimage.exe')
body = """
                <style>
                    table{
                        border: 1px solid black;
                    }
                </style>
                <table>
                <tr><td>Dit is een test</td><td>Blabla</td></tr>
                <tr><td>Dit is een test</td><td>Blabla</td></tr>
                </table>"""
imgkit.from_string(body, 'out.jpg', config=config)