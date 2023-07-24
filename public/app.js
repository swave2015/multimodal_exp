const socket = io();

socket.on("message", (message) => {
  const messages = document.getElementById("messages");
  const newMessage = document.createElement("div");
  newMessage.textContent = message;
  messages.appendChild(newMessage);
});

function sendMessage() {
  const input = document.getElementById("input");
  const message = input.value;
  input.value = "";

  if (message.trim()) {
    socket.emit("message", message);
  }
}
