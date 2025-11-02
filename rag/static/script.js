const API_URL = "http://localhost:8000/api/chat";

async function sendMessage() {
  const input = document.getElementById('userInput');
  const text = input.value.trim();
  if (!text) return;

  addMessage(text, 'user');
  input.value = '';

  try {
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: text })
    });

    if (!response.ok) throw new Error('Network response was not ok');

    const data = await response.json();
    addMessage(data.answer || '⚠️ لم يتم العثور على إجابة', 'bot');
  } catch (e) {
    addMessage('⚠️ حدث خطأ أثناء الاتصال بالخادم', 'bot');
    console.error(e);
  }
}

function addMessage(text, sender) {
  const box = document.getElementById('chat-box');
  const msg = document.createElement('div');
  msg.className = `msg ${sender}`;
  msg.innerText = text;
  box.appendChild(msg);
  box.scrollTop = box.scrollHeight;
}

document.getElementById('sendBtn').addEventListener('click', sendMessage);
document.getElementById('userInput').addEventListener('keypress', (e) => {
  if (e.key === 'Enter') sendMessage();
});

