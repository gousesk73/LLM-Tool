import React, { useState, useEffect } from "react";
import {
  Box,
  TextField,
  IconButton,
  Paper,
  Typography,
  CircularProgress,
} from "@mui/material";
import SendIcon from "@mui/icons-material/Send";
import ThumbUpIcon from "@mui/icons-material/ThumbUp";
import ThumbDownIcon from "@mui/icons-material/ThumbDown";
import axios from "axios";
import { toast } from "react-toastify";

const MainContent = ({
  apiKey,
  selectedModel,
  storeId,
  conversationId,
  chatHistory = [],
  setChatHistory,
}) => {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [feedbackLoading, setFeedbackLoading] = useState(false);

  useEffect(() => {
    if (!apiKey) console.error("API Key is missing");
    if (!selectedModel) console.error("Model is not selected");
    if (!storeId) console.error("Store ID is missing");
    if (!conversationId) console.error("Conversation ID is missing");
  }, [apiKey, selectedModel, storeId, conversationId]);

  const handleSend = async () => {
    if (!apiKey || !selectedModel || !storeId || !conversationId) {
      toast.error(
        "Please ensure you have provided an API key, selected a model, uploaded documents, and selected a conversation."
      );
      return;
    }

    try {
      setLoading(true);

      // Save user question in chat history before getting bot response
      const userMessage = { sender: "user", text: question };
      setChatHistory([...chatHistory, userMessage]);

      const formData = new FormData();
      formData.append("store_id", storeId);
      formData.append("user_question", question);
      formData.append("user_api_key", apiKey);
      formData.append("model_name", selectedModel);
      formData.append("conversation_id", conversationId);

      const response = await axios.post(
        "http://localhost:8000/ask-question/",
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
        }
      );

      const botResponse = response.data.response;
      const botMessageId = response.data.message_id; // Ensure backend sends this
      console.log("Bot Message ID:", botMessageId); // Debugging

      const newMessages = [
        userMessage,
        { sender: "bot", text: botResponse, messageId: botMessageId },
      ];

      setChatHistory([...chatHistory, ...newMessages]);
      setQuestion(""); // Reset input field
    } catch (error) {
      console.error("Error asking question:", error);
      toast.error("Error asking question");
    } finally {
      setLoading(false);
    }
  };

  const sendFeedback = async (messageId, feedback) => {
    if (!messageId) {
      console.error("Missing messageId for feedback.");
      toast.error("Unable to send feedback. Missing message ID.");
      return;
    }

    try {
      console.log("Sending Feedback:", { messageId, feedback }); // Debugging payload
      setFeedbackLoading(true); // Set loading state

      const response = await axios.post(
        "http://localhost:8000/feedback/",
        { messageId, feedback },
        { headers: { "Content-Type": "application/json" } }
      );

      toast.success(response.data.message);
      console.log("Feedback Response:", response.data); // Debug response
    } catch (error) {
      console.error(
        "Error sending feedback:",
        error.response?.data || error.message
      );
      toast.error("Failed to send feedback");
    } finally {
      setFeedbackLoading(false); // Reset loading state
    }
  };

  return (
    <Box sx={{ height: "97vh", display: "flex", flexDirection: "column" }}>
      <Box
        sx={{
          p: 1,
          borderBottom: "1px solid #ddd",
          display: "flex",
          alignItems: "center",
        }}
      >
        <Typography variant="h6">
          {selectedModel ? `Asking ${selectedModel}` : "Please select a model"}
        </Typography>
      </Box>
      <Box sx={{ flex: 1, overflowY: "auto", p: 2 }}>
        {chatHistory.map((message, index) => (
          <Box
            key={index}
            sx={{
              display: "flex",
              justifyContent:
                message.sender === "user" ? "flex-end" : "flex-start",
              mb: 1,
            }}
          >
            <Paper
              sx={{
                backgroundColor:
                  message.sender === "user" ? "#DCF8C6" : "#E5E5EA",
                color: "#000",
                borderRadius: 2,
                padding: "8px 12px",
                maxWidth: "70%",
                wordWrap: "break-word",
              }}
            >
              <Typography variant="body1" sx={{ whiteSpace: "pre-wrap" }}>
                {message.text}
              </Typography>
              {message.sender === "bot" && message.messageId && (
                <Box
                  sx={{
                    display: "flex",
                    justifyContent: "flex-end",
                    alignItems: "center",
                    mt: 1,
                  }}
                >
                  <IconButton
                    onClick={() => sendFeedback(message.messageId, "positive")}
                    disabled={feedbackLoading}
                  >
                    {feedbackLoading ? (
                      <CircularProgress size={20} />
                    ) : (
                      <ThumbUpIcon />
                    )}
                  </IconButton>
                  <IconButton
                    onClick={() => sendFeedback(message.messageId, "negative")}
                    disabled={feedbackLoading}
                  >
                    {feedbackLoading ? (
                      <CircularProgress size={20} />
                    ) : (
                      <ThumbDownIcon />
                    )}
                  </IconButton>
                </Box>
              )}
            </Paper>
          </Box>
        ))}
      </Box>
      <Box
        sx={{
          p: 1,
          backgroundColor: "#fff",
          borderTop: "1px solid #ddd",
          display: "flex",
          alignItems: "center",
          position: "sticky",
          bottom: 0,
          zIndex: 1,
        }}
      >
        <TextField
          fullWidth
          placeholder="Ask your question here..."
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyPress={(e) => {
            if (e.key === "Enter" && !loading && question.trim()) {
              handleSend();
            }
          }}
          InputProps={{
            endAdornment: (
              <IconButton
                onClick={handleSend}
                disabled={loading || !question.trim()}
                sx={{
                  backgroundColor: loading ? "transparent" : "#1976d2",
                  "&:hover": {
                    backgroundColor: loading ? "transparent" : "#1565c0",
                  },
                }}
              >
                {loading ? (
                  <CircularProgress size={24} />
                ) : (
                  <SendIcon sx={{ color: "#fff" }} />
                )}
              </IconButton>
            ),
          }}
          sx={{ mr: 1, maxWidth: "calc(100% - 60px)" }}
        />
      </Box>
    </Box>
  );
};

export default MainContent;
