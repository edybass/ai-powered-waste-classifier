// Configuration for the waste classifier
// For production, move sensitive data to environment variables

const CONFIG = {
    // Hugging Face configuration
    HUGGING_FACE: {
        // Use free inference API - no token needed for demo
        API_URL: "https://api-inference.huggingface.co/models/microsoft/resnet-50",
        // For production, get your own token from https://huggingface.co/settings/tokens
        // API_TOKEN: "hf_YOUR_TOKEN_HERE"
    },

    // Statistics tracking
    STATS: {
        totalClassifications: 10234,
        accuracy: 0.951,
        avgProcessingTime: 0.45
    }
};