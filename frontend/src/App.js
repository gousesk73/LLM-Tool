import React, { useState } from "react";
import { Container, Grid } from "@mui/material";
import Sidebar from "./components/Sidebar";
import MainContent from "./components/MainContent";
import { ToastContainer } from "react-toastify";
import 'react-toastify/dist/ReactToastify.css';

const App = () => {
  const [storeId, setStoreId] = useState(localStorage.getItem('storeId')); // Check localStorage for the storeId
  const [apiKey, setApiKey] = useState('');
  const [selectedModel, setSelectedModel] = useState('');
  const [chatHistory, setChatHistory] = useState([]);  // Define chatHistory and setChatHistory
  const [conversationId, setConversationId] = useState(null);  // New state to hold active conversationId

  return (
    <Container maxWidth="xl">
      <ToastContainer />
      <Grid container spacing={2}>
        <Grid item xs={3}>
          <Sidebar
            setStoreId={setStoreId}
            setApiKey={setApiKey}
            setSelectedModel={setSelectedModel}
            setChatHistory={setChatHistory}  // Pass setChatHistory as a prop
            setConversationId={setConversationId}  // Pass setConversationId to Sidebar
            apiKey={apiKey}  // Pass apiKey as a prop
            selectedModel={selectedModel}  // Pass selectedModel as a prop
            storeId={storeId}  // Pass storeId to Sidebar
          />
        </Grid>
        <Grid item xs={9}>
          <MainContent
            storeId={storeId}
            apiKey={apiKey}
            selectedModel={selectedModel}
            chatHistory={chatHistory}  // Pass chatHistory
            setChatHistory={setChatHistory}  // Pass setChatHistory as a prop
            conversationId={conversationId}  // Pass conversationId to MainContent
          />
        </Grid>
      </Grid>
    </Container>
  );
};

export default App;
