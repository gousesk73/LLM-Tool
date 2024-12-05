import React, { useState, useEffect } from "react";
import {
  Drawer,
  List,
  ListItem,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Typography,
  CircularProgress,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  IconButton,
} from "@mui/material";
import { toast } from "react-toastify";
import DeleteIcon from "@mui/icons-material/Delete";
import axios from "axios";

const Sidebar = ({
  setStoreId,
  setApiKey,
  setSelectedModel,
  storeId,
  setChatHistory,
  setConversationId,
  apiKey,
  selectedModel,
}) => {
  const [loading, setLoading] = useState(false);
  const [conversations, setConversations] = useState([]);
  const [activeConversationId, setActiveConversationId] = useState(null);
  const [openDialog, setOpenDialog] = useState(false);
  const [newChatTitle, setNewChatTitle] = useState("");

  // Fetch existing conversations on load
  useEffect(() => {
    const fetchConversations = async () => {
      try {
        const response = await axios.get(
          "http://localhost:8000/get-all-conversations/"
        );
        setConversations(response.data);
      } catch (error) {
        console.error("Error loading conversations:", error);
      }
    };

    fetchConversations();
  }, []);

  // Fetch the saved API key and model from the backend on load
  useEffect(() => {
    const fetchApiKeyAndModel = async () => {
      try {
        const response = await axios.get(
          "http://localhost:8000/get-api-key-and-model/"
        );
        setApiKey(response.data.api_key);
        setSelectedModel(response.data.model_name);
      } catch (error) {
        console.error("Error fetching API key and model:", error);
      }
    };

    fetchApiKeyAndModel();
  }, [setApiKey, setSelectedModel]);

  // Check localStorage for storeId on component mount
  useEffect(() => {
    const savedStoreId = localStorage.getItem("storeId");
    if (savedStoreId) {
      setStoreId(savedStoreId);
    }
  }, [setStoreId]);

  const handleDeleteConversation = async (conversationId) => {
    const confirmDelete = window.confirm(
      "Are you sure you want to delete this conversation?"
    );
    if (confirmDelete) {
      try {
        // Call the backend delete endpoint
        await axios.delete(
          `http://localhost:8000/delete-conversation/${conversationId}`
        );

        // Refresh the conversation list
        const updatedConversations = await axios.get(
          "http://localhost:8000/get-all-conversations/"
        );
        setConversations(updatedConversations.data);

        // Show success message
        toast.success("Conversation deleted.");

        // Clear chat history
        setChatHistory([]);
        setActiveConversationId(null);
      } catch (error) {
        console.error("Error deleting conversation:", error);
        toast.error("Failed to delete conversation");
      }
    }
  };

  // Create new conversation logic
  const handleNewChat = async () => {
    setOpenDialog(false);

    // Ensure storeId is present before creating a chat
    if (!storeId) {
      toast.error(
        "Please upload PDFs to generate a store ID before creating a new chat."
      );
      return;
    }

    try {
      const formData = new FormData();
      formData.append("title", newChatTitle);
      formData.append("store_id", storeId); // Pass the store_id generated from the document upload

      const response = await axios.post(
        "http://localhost:8000/save-conversation/",
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
        }
      );

      const conversationId = response.data.conversation_id;
      setChatHistory([]); // Clear previous chat history
      setActiveConversationId(conversationId); // Set the newly created conversation as active
      setConversationId(conversationId); // Update the parent state with the new conversationId

      // Fetch updated conversations list
      const updatedConversations = await axios.get(
        "http://localhost:8000/get-all-conversations/"
      );
      setConversations(updatedConversations.data);

      toast.success("New conversation created");
    } catch (error) {
      console.error("Error creating new conversation:", error);
      toast.error("Failed to create new conversation");
    }
  };

  // Function to load a conversation and its messages
  const loadConversation = async (conversation) => {
    try {
      const formData = new FormData();
      formData.append("conversation_id", conversation.conversation_id);

      // Fetch both conversation messages and storeId from the backend
      const response = await axios.post(
        "http://localhost:8000/get-conversation-messages/",
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
        }
      );

      // Process messages to ensure sender information is correctly set
      const loadedMessages = response.data.messages.map((msg) => ({
        ...msg,
        sender: msg.sender || "bot", // Set 'bot' if sender is not specified
      }));

      // Set the conversation messages, storeId, and conversationId
      setChatHistory(loadedMessages);
      setActiveConversationId(conversation.conversation_id);
      setConversationId(conversation.conversation_id);

      // Set storeId from the fetched conversation
      setStoreId(response.data.store_id);

      // Optionally save conversationId and storeId to localStorage (if you want them to persist across reloads)
      localStorage.setItem("conversationId", conversation.conversation_id);
      localStorage.setItem("storeId", response.data.store_id);
    } catch (error) {
      console.error("Error loading conversation messages:", error);
      toast.error("Failed to load conversation messages.");
    }
  };

  return (
    <Drawer
      variant="permanent"
      anchor="left"
      sx={{
        width: 240,
        flexShrink: 0,
        "& .MuiDrawer-paper": { width: 240, boxSizing: "border-box" },
      }}
    >
      <List>
        <ListItem>
          <Typography variant="h6" noWrap>
            Options
          </Typography>
        </ListItem>

        <ListItem>
          <Button variant="contained" component="label" sx={{ width: "100%" }}>
            Upload Folders
            <input
              type="file"
              webkitdirectory="true"
              directory="true"
              multiple
              hidden
              onChange={async (event) => {
                const folderFiles = Array.from(event.target.files);
                const formData = new FormData();
                folderFiles.forEach((file) => formData.append("files", file));
                try {
                  setLoading(true);
                  const response = await axios.post(
                    "http://localhost:8000/process-pdfs/",
                    formData
                  );
                  const newStoreId = response.data.store_id;
                  setStoreId(newStoreId);
                  localStorage.setItem("storeId", newStoreId); // Save storeId in localStorage
                  toast.success(response.data.message);
                } catch (error) {
                  toast.error("Error uploading PDFs");
                } finally {
                  setLoading(false);
                }
              }}
            />
          </Button>
        </ListItem>
        {loading && (
          <ListItem>
            <CircularProgress />
          </ListItem>
        )}

        {/* API Key and Model inputs visible in Sidebar */}
        <ListItem>
          <FormControl fullWidth>
            <InputLabel>Select AI Model</InputLabel>
            <Select
              value={selectedModel || ""}
              onChange={(e) => setSelectedModel(e.target.value)}
            >
              <MenuItem value="ChatGPT">ChatGPT</MenuItem>
              <MenuItem value="gemini-1.5-flash">GoogleGemini</MenuItem>
            </Select>
          </FormControl>
        </ListItem>

        <ListItem>
          <TextField
            fullWidth
            label="Enter API Key"
            value={apiKey || ""}
            onChange={(e) => setApiKey(e.target.value)}
          />
        </ListItem>

        <ListItem>
          <Button
            variant="contained"
            fullWidth
            onClick={async () => {
              const formData = new FormData();
              formData.append("model_name", selectedModel);
              formData.append("api_key", apiKey);
              try {
                await axios.post(
                  "http://localhost:8000/save-api-key/",
                  formData
                );
                toast.success("API Key submitted and saved!");
              } catch (error) {
                toast.error("Please enter a valid API key.");
              }
            }}
          >
            Submit API Key
          </Button>
        </ListItem>

        <ListItem>
          <Button
            variant="contained"
            fullWidth
            onClick={() => setOpenDialog(true)}
          >
            New Chat
          </Button>
        </ListItem>

        {conversations.map((conversation) => (
          <ListItem
            button
            key={conversation.conversation_id}
            onClick={() => loadConversation(conversation)}
            sx={{
              backgroundColor:
                activeConversationId === conversation.conversation_id
                  ? "#f0f0f0"
                  : "transparent",
            }}
          >
            <Typography sx={{ flexGrow: 1 }}>{conversation.title}</Typography>
            <IconButton
              onClick={() =>
                handleDeleteConversation(conversation.conversation_id)
              }
              aria-label="delete"
            >
              <DeleteIcon />
            </IconButton>
          </ListItem>
        ))}

        <Dialog open={openDialog} onClose={() => setOpenDialog(false)}>
          <DialogTitle>New Chat</DialogTitle>
          <DialogContent>
            <TextField
              autoFocus
              margin="dense"
              label="Chat Title"
              fullWidth
              value={newChatTitle}
              onChange={(e) => setNewChatTitle(e.target.value)}
            />
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setOpenDialog(false)}>Cancel</Button>
            <Button onClick={handleNewChat}>Create</Button>
          </DialogActions>
        </Dialog>
      </List>
    </Drawer>
  );
};

export default Sidebar;
