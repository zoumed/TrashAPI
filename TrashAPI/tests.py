import unittest
import json
from flask import Flask, request

from app import home



class TestHome(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__)
        self.client = self.app.test_client()
        self.app.testing = True
        with self.app.test_request_context(method='POST', data={'image': open('./images/icetea.jpeg', 'rb'), 'code': '75002'}):
            self.response = home()

    def test_home(self):
        # Vérifier si la réponse est un code HTTP 200 OK
        self.assertEqual(self.response.status_code, 200)
        # Vérifier si la réponse contient les données attendues
        data = json.loads(self.response.get_data(as_text=True))
        self.assertIn('colorTrash', data)
        self.assertIn('probability', data)
        self.assertIn('typeTrash', data)
        self.assertIn('adresses', data)
        # Vérifier si les données de la réponse sont correctes
        self.assertEqual(data['colorTrash'], 'yellow')
        self.assertGreater(float(data['probability']), 0.5)
        self.assertEqual(data['typeTrash'], 'recyclable')
        self.assertGreater(len(data['adresses']), 0)



class TestHome2(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__)
        self.client = self.app.test_client()
        self.app.testing = True
        with self.app.test_request_context(method='POST', data={'image': open('./images/ccc.jpeg', 'rb'), 'code': '75013'}):
            self.response = home()

    def test_home2(self):
        # Vérifier si la réponse est un code HTTP 200 OK
        self.assertEqual(self.response.status_code, 200)
        # Vérifier si la réponse contient les données attendues
        data = json.loads(self.response.get_data(as_text=True))
        self.assertIn('colorTrash', data)
        self.assertIn('probability', data)
        self.assertIn('typeTrash', data)
        self.assertIn('adresses', data)
        # Vérifier si les données de la réponse sont correctes
        self.assertEqual(data['colorTrash'], 'black')
        self.assertGreater(float(data['probability']), 0.5)
        self.assertEqual(data['typeTrash'], 'organic')
        self.assertEqual(data['adresses'], [
        "22 rue Pierre Gourdault",
        "20 rue Saillard",
        "Face au 23 Villa d'Este"
    ])
