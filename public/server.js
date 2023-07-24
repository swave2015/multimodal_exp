const express = require('express');
const app = express();
const http = require('http').Server(app);
const io = require('socket.io')(http);
const path = require('path');

// Serve static files from the 'public' folder
app.use(express.static(path.join(__dirname, 'public')));

// Define a route for the root path
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

// Your Socket.IO code for handling chat events goes here

// Start the server
const port = process.env.PORT || 3000;
http.listen(port, () => {
  console.log(`Server running on port ${port}`);
});
