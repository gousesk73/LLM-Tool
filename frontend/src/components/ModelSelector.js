import React, { useState } from 'react';
import { Box, FormControl, InputLabel, Select, MenuItem, TextField, Button } from '@mui/material';
import axios from 'axios';
import { toast } from 'react-toastify';

const ModelSelector = ({ setSelectedModel, setApiKey }) => {
  const [apiInput, setApiInput] = useState("");
  const [model, setModel] = useState("");
  const [showApiKeyInput, setShowApiKeyInput] = useState(false);

  const handleModelChange = (event) => {
    const selectedModel = event.target.value;
    setModel(selectedModel);
    setSelectedModel(selectedModel); // Pass selected model to parent
    setShowApiKeyInput(true); // Show API key input when model is selected
  };

  const handleApiKeyChange = async () => {
    const formData = new FormData();
    formData.append('model_name', model);
    formData.append('api_key', apiInput);

    try {
      await axios.post('http://localhost:8000/save-api-key/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setApiKey(apiInput); // Set API key in parent component
      toast.success(`API Key for ${model} submitted and saved.`);
    } catch (error) {
      console.error('Error saving API key:', error);
      toast.error('Failed to save API key.');
    }
  };

  return (
    <Box sx={{ mb: 4 }}>
      <FormControl fullWidth sx={{ mb: 2 }}>
        <InputLabel>Select AI Model</InputLabel>
        <Select
          value={model}
          label="Select AI Model"
          onChange={handleModelChange}
        >
          <MenuItem value="ChatGPT">ChatGPT</MenuItem>
          <MenuItem value="GoogleGemini">Google Gemini</MenuItem>
        </Select>
      </FormControl>

      {showApiKeyInput && (
        <Box>
          <TextField
            fullWidth
            label="Enter API Key"
            value={apiInput}
            onChange={(e) => setApiInput(e.target.value)}
            sx={{ mb: 2 }}
          />
          <Button variant="contained" onClick={handleApiKeyChange}>
            Submit API Key
          </Button>
        </Box>
      )}
    </Box>
  );
};

export default ModelSelector;
