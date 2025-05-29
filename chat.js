// server.js
const express = require('express');
const axios = require('axios');
const cors = require('cors');
const app = express();
app.use(cors());
app.use(express.json());

const OPENAI_API_KEY = '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';

app.post('/chat', async (req, res) => {
  try {
    const { message } = req.body;
    const response = await axios.post(
      'https://api.openai.com/v1/chat/completions',
      {
        model: 'gpt-3.5-turbo',
        messages: [
          { role: 'system', content: 'You are a helpful health assistant.' },
          { role: 'user', content: message }
        ],
        temperature: 0.2
      },
      {
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${OPENAI_API_KEY}`
        }
      }
    );
    res.json({ reply: response.data.choices[0].message.content });
  } catch (err) {
    res.status(500).json({ error: 'Failed to get response from OpenAI.' });
  }
});

app.listen(3001, () => console.log('Backend running on port 3001'));
