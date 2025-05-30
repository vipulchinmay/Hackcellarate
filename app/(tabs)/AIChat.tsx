import React, { useState, useRef, useEffect } from 'react';
import {
  View,
  Text,
  TextInput,
  FlatList,
  TouchableOpacity,
  StyleSheet,
  SafeAreaView,
  KeyboardAvoidingView,
  Platform,
  ActivityIndicator,
  Alert,
  Animated,
} from 'react-native';
import { StatusBar } from 'expo-status-bar';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import { LinearGradient } from 'expo-linear-gradient';
import { Audio } from 'expo-av';
import * as Speech from 'expo-speech';
import { useLanguage } from '@/components/LanguageContext';
import translations from '@/components/translation';

// Replace with your Gemini API key
const GEMINI_API_KEY = 'AIzaSyB91pjFVJq3NbLfU0sV4dQPnkl3DQSCqRo';
const GEMINI_API_URL = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent';

const MEDICAL_KNOWLEDGE_BASE = {
  // Common Symptoms with detailed keywords
  symptoms: {
    pain: {
      keywords: ['pain', 'ache', 'hurt', 'hurting', 'sore', 'tender', 'burning', 'sharp', 'dull', 'throbbing', 'stabbing', 'cramping', 'stinging'],
      bodyParts: ['head', 'back', 'chest', 'stomach', 'abdomen', 'leg', 'arm', 'neck', 'shoulder', 'knee', 'ankle', 'wrist', 'joint'],
      severity: ['mild', 'moderate', 'severe', 'intense', 'unbearable', 'chronic', 'acute']
    },
    respiratory: {
      keywords: ['cough', 'coughing', 'sneeze', 'sneezing', 'wheeze', 'wheezing', 'breathless', 'shortness of breath', 'congestion', 'stuffy nose', 'runny nose', 'phlegm', 'mucus', 'sputum'],
      types: ['dry cough', 'wet cough', 'persistent cough', 'chronic cough', 'productive cough']
    },
    gastrointestinal: {
      keywords: ['nausea', 'vomiting', 'diarrhea', 'diarrhoea', 'constipation', 'bloating', 'gas', 'indigestion', 'heartburn', 'acid reflux', 'stomach ache', 'cramps', 'loose stools', 'upset stomach', 'food poisoning'],
      severity: ['mild', 'severe', 'chronic', 'occasional', 'frequent']
    },
    neurological: {
      keywords: ['headache', 'migraine', 'dizziness', 'dizzy', 'vertigo', 'lightheaded', 'confusion', 'memory loss', 'concentration', 'focus', 'numbness', 'tingling', 'seizure'],
      types: ['tension headache', 'cluster headache', 'sinus headache']
    },
    systemic: {
      keywords: ['fever', 'temperature', 'chills', 'fatigue', 'tired', 'weakness', 'exhaustion', 'malaise', 'body aches', 'sweating', 'night sweats', 'weight loss', 'weight gain', 'appetite loss']
    },
    dermatological: {
      keywords: ['rash', 'itching', 'itchy', 'hives', 'swelling', 'redness', 'inflammation', 'bruising', 'cuts', 'wounds', 'burns', 'dry skin', 'acne', 'eczema', 'psoriasis']
    },
    psychological: {
      keywords: ['stress', 'anxiety', 'worried', 'depression', 'sad', 'mood', 'panic', 'fear', 'nervous', 'overwhelmed', 'burnout', 'insomnia', 'sleep problems', 'nightmares']
    }
  },

  // Common Medical Conditions
  conditions: {
    infectious: {
      keywords: ['cold', 'flu', 'covid', 'coronavirus', 'infection', 'virus', 'bacterial', 'pneumonia', 'bronchitis', 'sinusitis', 'UTI', 'urinary tract infection'],
      symptoms: ['fever', 'cough', 'congestion', 'body aches', 'fatigue']
    },
    chronic: {
      keywords: ['diabetes', 'hypertension', 'high blood pressure', 'asthma', 'arthritis', 'fibromyalgia', 'chronic pain', 'IBS', 'irritable bowel syndrome', 'GERD', 'acid reflux'],
      management: ['medication', 'lifestyle changes', 'diet', 'exercise']
    },
    cardiovascular: {
      keywords: ['heart', 'chest pain', 'palpitations', 'irregular heartbeat', 'high cholesterol', 'blood pressure', 'circulation', 'shortness of breath']
    },
    metabolic: {
      keywords: ['diabetes', 'blood sugar', 'glucose', 'insulin', 'metabolism', 'thyroid', 'hormone', 'endocrine']
    }
  },

  // Treatment and Management Options
  treatments: {
    medications: {
      keywords: ['medicine', 'medication', 'pills', 'tablets', 'prescription', 'over-the-counter', 'OTC', 'antibiotics', 'painkillers', 'ibuprofen', 'acetaminophen', 'aspirin'],
      types: ['oral', 'topical', 'injection', 'inhaler']
    },
    natural: {
      keywords: ['home remedy', 'natural treatment', 'herbal', 'alternative medicine', 'acupuncture', 'massage', 'yoga', 'meditation', 'essential oils'],
      methods: ['rest', 'hydration', 'heat therapy', 'cold therapy', 'breathing exercises']
    },
    lifestyle: {
      keywords: ['diet', 'exercise', 'sleep', 'stress management', 'lifestyle changes', 'nutrition', 'physical therapy', 'rehabilitation']
    }
  }
};

const AIChat = () => {
  const [messages, setMessages] = useState([
    {
      id: '1',
      text: "Hello! I'm your AushadX AI assistant. I can help you with medical questions, general health advice, or just have a friendly conversation. What's on your mind today?",
      sender: 'bot',
      timestamp: new Date(),
    },
  ]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [recording, setRecording] = useState(null);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [typingIndicator, setTypingIndicator] = useState(false);
  const [conversationContext, setConversationContext] = useState([]);
  const { language } = useLanguage();
  const navigation = useNavigation();
  const flatListRef = useRef(null);
  const typingAnimation = useRef(new Animated.Value(0)).current;

  const t = (key) => translations[language]?.[key] || key;

  useEffect(() => {
    if (flatListRef.current && messages.length > 0) {
      setTimeout(() => {
        flatListRef.current.scrollToEnd({ animated: true });
      }, 100);
    }
  }, [messages]);

  // Enhanced conversation context management
  const updateConversationContext = (userMessage, botResponse) => {
    const newContext = {
      user: userMessage,
      bot: botResponse,
      timestamp: new Date(),
    };
    
    setConversationContext(prev => {
      const updated = [...prev, newContext];
      // Keep last 5 exchanges for context
      return updated.slice(-5);
    });
  };

  // Enhanced natural language processing
const analyzeMessage = (message) => {
  const lowerMessage = message.toLowerCase().trim();
  
  const analysis = {
    type: 'general',
    intent: 'conversation',
    entities: [],
    symptoms: [],
    conditions: [],
    bodyParts: [],
    treatments: [],
    severity: null,
    urgency: 'low',
    medicalRelevance: 0,
    sentiment: 'neutral',
    categories: []
  };

  let medicalScore = 0;

  // Analyze symptoms
  Object.entries(MEDICAL_KNOWLEDGE_BASE.symptoms).forEach(([category, data]) => {
    data.keywords.forEach(keyword => {
      if (lowerMessage.includes(keyword)) {
        analysis.symptoms.push({ keyword, category });
        analysis.entities.push(keyword);
        analysis.categories.push(category);
        medicalScore += 1;
      }
    });

    // Check for body parts if it's pain-related
    if (category === 'pain' && data.bodyParts) {
      data.bodyParts.forEach(bodyPart => {
        if (lowerMessage.includes(bodyPart)) {
          analysis.bodyParts.push(bodyPart);
          medicalScore += 0.5;
        }
      });
    }

    // Check severity indicators
    if (data.severity) {
      data.severity.forEach(level => {
        if (lowerMessage.includes(level)) {
          analysis.severity = level;
          medicalScore += 0.5;
        }
      });
    }
  });

  // Analyze conditions
  Object.entries(MEDICAL_KNOWLEDGE_BASE.conditions).forEach(([category, data]) => {
    data.keywords.forEach(keyword => {
      if (lowerMessage.includes(keyword)) {
        analysis.conditions.push({ keyword, category });
        analysis.entities.push(keyword);
        medicalScore += 1.5; // Conditions get higher weight
      }
    });
  });

  // Analyze treatments mentioned
  Object.entries(MEDICAL_KNOWLEDGE_BASE.treatments).forEach(([category, data]) => {
    data.keywords.forEach(keyword => {
      if (lowerMessage.includes(keyword)) {
        analysis.treatments.push({ keyword, category });
        medicalScore += 0.5;
      }
    });
  });

  // Determine urgency based on keywords
  const urgentKeywords = ['severe', 'intense', 'unbearable', 'emergency', 'urgent', 'can\'t breathe', 'chest pain', 'bleeding', 'unconscious'];
  const moderateKeywords = ['persistent', 'chronic', 'worsening', 'getting worse', 'not improving'];
  
  if (urgentKeywords.some(keyword => lowerMessage.includes(keyword))) {
    analysis.urgency = 'high';
    medicalScore += 2;
  } else if (moderateKeywords.some(keyword => lowerMessage.includes(keyword))) {
    analysis.urgency = 'moderate';
    medicalScore += 1;
  }

  // Calculate medical relevance (0-1 scale)
  analysis.medicalRelevance = Math.min(medicalScore / 5, 1);

  // Determine intent
  const questionWords = ['what', 'how', 'why', 'when', 'where', 'which', 'who', 'can', 'should', 'will', 'is', 'are', 'do', 'does'];
  const hasQuestionWord = questionWords.some(word => lowerMessage.startsWith(word));
  const hasQuestionMark = message.includes('?');
  
  if (hasQuestionWord || hasQuestionMark) {
    analysis.intent = 'question';
  }

  // Detect greetings
  const greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening', 'how are you'];
  if (greetings.some(greeting => lowerMessage.includes(greeting))) {
    analysis.intent = 'greeting';
  }

  // Sentiment analysis
  const positiveWords = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'happy', 'pleased', 'thank', 'better', 'improved'];
  const negativeWords = ['bad', 'terrible', 'awful', 'worried', 'concerned', 'scared', 'pain', 'hurt', 'sick', 'worse', 'horrible'];
  
  const positiveCount = positiveWords.filter(word => lowerMessage.includes(word)).length;
  const negativeCount = negativeWords.filter(word => lowerMessage.includes(word)).length;
  
  if (positiveCount > negativeCount) analysis.sentiment = 'positive';
  else if (negativeCount > positiveCount) analysis.sentiment = 'negative';

  return analysis;
};

  // Enhanced response generation
  // Replace your existing generateResponse function with this:
const generateResponse = async (userMessage, analysis) => {
  try {
    // Handle different types of conversations
    if (analysis.intent === 'greeting') {
      return generateGreetingResponse(analysis.sentiment);
    }

    // Enhanced medical relevance check - now more sensitive
    if (analysis.medicalRelevance > 0.1) { // Lowered threshold from 0.3 to 0.1
      return generateMedicalResponse(userMessage, analysis);
    }

    return generateGeneralResponse(userMessage, analysis);

  } catch (error) {
    console.error('Error generating response:', error);
    return "I'm having trouble processing that right now. Could you try rephrasing your question?";
  }
};

  const generateGreetingResponse = (sentiment) => {
    const greetings = [
      "Hello there! I'm doing well, thank you for asking. How can I assist you today?",
      "Hi! Great to hear from you. What would you like to talk about?",
      "Hey! I'm here and ready to help. What's on your mind?",
      "Good to see you! How are you feeling today? Is there anything I can help you with?"
    ];
    
    return greetings[Math.floor(Math.random() * greetings.length)];
  };

  const generateMedicalResponse = (message, analysis) => {
  const lowerMessage = message.toLowerCase();
  
  // Handle high urgency cases
  if (analysis.urgency === 'high') {
    return `I'm concerned about the symptoms you're describing. ${getUrgencyResponse(analysis)} Please seek immediate medical attention or contact emergency services if this is severe. In the meantime, ${getImmediateCareAdvice(analysis)}`;
  }

  // Handle specific symptom combinations
  if (analysis.symptoms.length > 0) {
    let response = getSymptomSpecificResponse(analysis);
    
    // Add severity-based advice
    if (analysis.severity) {
      response += ` ${getSeverityAdvice(analysis.severity)}`;
    }
    
    // Add general care advice
    response += ` ${getGeneralCareAdvice(analysis)}`;
    
    // Add when to seek help
    response += ` ${getSeekHelpAdvice(analysis)}`;
    
    return response;
  }

  // Handle condition-specific queries
  if (analysis.conditions.length > 0) {
    return getConditionSpecificResponse(analysis);
  }

  // Handle treatment queries
  if (analysis.treatments.length > 0) {
    return getTreatmentResponse(analysis);
  }

  // Generic medical response
  return "I understand you have a health-related question. Could you provide more specific details about your symptoms or concerns so I can offer more targeted guidance? Remember, while I can provide general health information, it's always best to consult with a healthcare professional for personalized medical advice.";
};

const getSymptomSpecificResponse = (analysis) => {
  const primarySymptoms = analysis.symptoms.map(s => s.keyword);
  const categories = [...new Set(analysis.categories)];
  
  if (categories.includes('respiratory')) {
    if (primarySymptoms.some(s => ['cough', 'coughing'].includes(s))) {
      return "I understand you're dealing with a cough. Coughs can be caused by various factors including viral infections, allergies, or environmental irritants.";
    }
  }
  
  if (categories.includes('gastrointestinal')) {
    if (primarySymptoms.some(s => ['diarrhea', 'diarrhoea', 'loose stools'].includes(s))) {
      return "I see you're experiencing digestive issues. Diarrhea can be caused by infections, food intolerances, stress, or medications.";
    }
    if (primarySymptoms.some(s => ['nausea', 'vomiting'].includes(s))) {
      return "Nausea and vomiting can be uncomfortable and may result from various causes including food poisoning, medications, or viral infections.";
    }
  }
  
  if (categories.includes('neurological')) {
    if (primarySymptoms.some(s => ['headache', 'migraine'].includes(s))) {
      return "Headaches can significantly impact your daily life. They may be triggered by stress, dehydration, eye strain, lack of sleep, or certain foods.";
    }
  }
  
  if (categories.includes('systemic')) {
    if (primarySymptoms.some(s => ['fever', 'temperature'].includes(s))) {
      return "A fever indicates your body is fighting an infection or inflammation. It's your immune system's natural response.";
    }
  }
  
  return `I understand you're experiencing ${primarySymptoms.join(', ')}. These symptoms can have various underlying causes.`;
};

const getGeneralCareAdvice = (analysis) => {
  const categories = [...new Set(analysis.categories)];
  let advice = [];
  
  if (categories.includes('respiratory')) {
    advice.push("Stay hydrated, use a humidifier, and consider warm saltwater gargles");
  }
  
  if (categories.includes('gastrointestinal')) {
    advice.push("Focus on staying hydrated with clear fluids, consider the BRAT diet (bananas, rice, applesauce, toast), and avoid dairy and fatty foods temporarily");
  }
  
  if (categories.includes('systemic')) {
    advice.push("Get plenty of rest, stay hydrated, and monitor your temperature");
  }
  
  if (categories.includes('neurological')) {
    advice.push("Rest in a quiet, dark room, apply a cold or warm compress, and ensure you're well-hydrated");
  }
  
  if (advice.length === 0) {
    advice.push("Focus on rest, proper hydration, and maintaining good nutrition");
  }
  
  return `General care recommendations include: ${advice.join(', ')}.`;
};

const getSeverityAdvice = (severity) => {
  switch (severity) {
    case 'mild':
      return "Since this seems mild, home care measures may be sufficient initially.";
    case 'moderate':
      return "Given the moderate nature of your symptoms, monitoring and appropriate care are important.";
    case 'severe':
    case 'intense':
    case 'unbearable':
      return "Given the severity you're describing, this warrants prompt medical evaluation.";
    case 'chronic':
      return "For chronic symptoms, consistent management and regular medical follow-up are important.";
    default:
      return "";
  }
};

const getSeekHelpAdvice = (analysis) => {
  if (analysis.urgency === 'high') {
    return "Please seek immediate medical attention.";
  }
  
  const redFlags = [
    "If symptoms worsen or persist for more than a few days",
    "If you develop a high fever (over 103°F/39.4°C)",
    "If you experience difficulty breathing or severe pain",
    "If you have any concerning changes in your condition"
  ];
  
  return `Consider consulting a healthcare provider if: ${redFlags.slice(0, 2).join(', ')}.`;
};

const getConditionSpecificResponse = (analysis) => {
  const condition = analysis.conditions[0];
  
  // Add comprehensive condition-specific responses here
  const conditionResponses = {
    'cold': "Common colds are viral infections that typically resolve on their own within 7-10 days. Focus on rest, hydration, and symptom management.",
    'flu': "Influenza is a viral infection that can be more severe than a cold. Antiviral medications may help if started early. Rest and hydration are crucial.",
    'diabetes': "Diabetes management involves monitoring blood sugar, following a balanced diet, regular exercise, and taking medications as prescribed by your healthcare provider.",
    'hypertension': "High blood pressure management typically includes lifestyle modifications like diet changes, exercise, stress management, and medications when necessary."
  };
  
  return conditionResponses[condition.keyword] || `Regarding ${condition.keyword}, it's important to work with healthcare professionals for proper management and treatment.`;
};

const getTreatmentResponse = (analysis) => {
  return "Treatment options can vary significantly based on the specific condition and individual factors. It's best to discuss treatment options with a qualified healthcare provider who can evaluate your specific situation and medical history.";
};

const getUrgencyResponse = (analysis) => {
  return "Based on what you're describing, this could require immediate medical evaluation.";
};

const getImmediateCareAdvice = (analysis) => {
  return "try to stay calm and avoid activities that might worsen your symptoms.";
};

  const generateGeneralResponse = (message, analysis) => {
    const lowerMessage = message.toLowerCase();

    // Handle common conversational topics
    if (lowerMessage.includes('how are you') || lowerMessage.includes('how do you feel')) {
      return "Thank you for asking! I'm functioning well and ready to help. I enjoy our conversations and learning about different topics. How are you doing today?";
    }

    if (lowerMessage.includes('what can you do') || lowerMessage.includes('what are your capabilities')) {
      return "I can help with a variety of things! I'm particularly knowledgeable about health and medical topics, but I can also have general conversations, answer questions on various subjects, help with problem-solving, or just chat about whatever interests you. What would you like to explore today?";
    }

    if (lowerMessage.includes('thank you') || lowerMessage.includes('thanks')) {
      return "You're very welcome! I'm glad I could help. Is there anything else you'd like to discuss or any other questions I can answer for you?";
    }

    if (lowerMessage.includes('weather')) {
      return "I don't have access to current weather data, but I'd be happy to discuss general weather-related health topics, like how weather changes can affect our mood, or tips for staying healthy in different seasons. What weather-related topic interests you?";
    }

    if (lowerMessage.includes('joke') || lowerMessage.includes('funny')) {
      return "Here's a light-hearted one for you: Why did the doctor carry a red pen? In case they needed to draw blood! 😄 I hope that brought at least a small smile to your face. Laughter really can be good medicine sometimes!";
    }

    // Context-aware responses
    if (conversationContext.length > 0) {
      const lastContext = conversationContext[conversationContext.length - 1];
      if (lowerMessage.includes('more about') || lowerMessage.includes('tell me more')) {
        return "I'd be happy to elaborate on what we were discussing! Could you be more specific about which part you'd like me to expand on?";
      }
    }

    // Default conversational response
    const responses = [
      "That's interesting! Tell me more about what you're thinking.",
      "I appreciate you sharing that with me. What would you like to explore further?",
      "That's a good point. How do you feel about that topic?",
      "I find that perspective intriguing. What led you to think about this?",
      "Thank you for bringing that up. Is there a particular aspect you'd like to discuss more?"
    ];

    return responses[Math.floor(Math.random() * responses.length)];
  };

  // Enhanced message processing
  const sendMessageToGemini = async (userMessage) => {
    try {
      setIsLoading(true);
      setTypingIndicator(true);
      
      // Analyze the message
      const analysis = analyzeMessage(userMessage);
      
      // Simulate processing time with typing animation
      Animated.loop(
        Animated.sequence([
          Animated.timing(typingAnimation, {
            toValue: 1,
            duration: 500,
            useNativeDriver: true,
          }),
          Animated.timing(typingAnimation, {
            toValue: 0,
            duration: 500,
            useNativeDriver: true,
          }),
        ])
      ).start();

      // Simulate realistic response time
      await new Promise(resolve => setTimeout(resolve, 1500 + Math.random() * 1000));
      
      // Generate contextual response
      const response = await generateResponse(userMessage, analysis);
      
      // Update conversation context
      updateConversationContext(userMessage, response);
      
      return response;

    } catch (error) {
      console.error('Error processing message:', error);
      return "I apologize, but I'm having some technical difficulties right now. Please try again in a moment.";
    } finally {
      setIsLoading(false);
      setTypingIndicator(false);
      typingAnimation.stopAnimation();
    }
  };

  // Enhanced message handling
  const handleSendMessage = async () => {
    if (inputText.trim() === '' || isLoading) return;

    const userMessage = {
      id: Date.now().toString(),
      text: inputText.trim(),
      sender: 'user',
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputText('');

    // Get AI response
    const aiResponse = await sendMessageToGemini(userMessage.text);

    // Add AI response to chat
    const botMessage = {
      id: (Date.now() + 1).toString(),
      text: aiResponse,
      sender: 'bot',
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, botMessage]);
    
    // Read out the AI response
    speakMessage(aiResponse);
  };

  // Enhanced text-to-speech with better controls
  const speakMessage = (message) => {
    if (isSpeaking) {
      Speech.stop();
    }
    
    setIsSpeaking(true);
    
    // Clean message for better speech synthesis
    const cleanMessage = message.replace(/[😄🙂😊]/g, '').trim();
    
    Speech.speak(cleanMessage, {
      language: language === 'hi' ? 'hi-IN' : 'en-US',
      pitch: 1.0,
      rate: 0.85,
      volume: 1.0,
      onDone: () => setIsSpeaking(false),
      onError: () => setIsSpeaking(false),
      onStopped: () => setIsSpeaking(false),
    });
  };

  const stopSpeaking = () => {
    if (isSpeaking) {
      Speech.stop();
      setIsSpeaking(false);
    }
  };

  // Enhanced voice input with better sample queries
  const handleVoiceInput = async (recordingUri) => {
    setIsLoading(true);
    
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    const sampleQueries = [
      // Medical queries
      "I've been having headaches lately, what could be causing them?",
      "How can I improve my sleep quality?",
      "What are some natural ways to reduce stress?",
      "I'm feeling anxious about my health, can you help?",
      "What should I eat for better nutrition?",
      "How much exercise do I need each week?",
      "I have a persistent cough, should I be concerned?",
      "What are the symptoms of dehydration?",
      "How can I boost my immune system?",
      "I'm having trouble concentrating, any suggestions?",
      
      // Conversational queries
      "How are you doing today?",
      "What can you help me with?",
      "I'm feeling a bit down, can we chat?",
      "Tell me something interesting about health",
      "What's the best way to stay motivated?",
      "I'm bored, what should we talk about?",
      "Can you tell me a joke?",
      "What do you think about mental health?",
      "How important is work-life balance?",
      "What makes a person healthy and happy?"
    ];
    
    const randomQuery = sampleQueries[Math.floor(Math.random() * sampleQueries.length)];
    
    const userMessage = {
      id: Date.now().toString(),
      text: randomQuery,
      sender: 'user',
      timestamp: new Date(),
    };
    
    setMessages((prev) => [...prev, userMessage]);
    
    const aiResponse = await sendMessageToGemini(randomQuery);
    
    const botMessage = {
      id: (Date.now() + 1).toString(),
      text: aiResponse,
      sender: 'bot',
      timestamp: new Date(),
    };
    
    setMessages((prev) => [...prev, botMessage]);
    setIsLoading(false);
    
    speakMessage(aiResponse);
  };

  // Voice recording functions (unchanged)
  const startRecording = async () => {
    try {
      if (isSpeaking) {
        stopSpeaking();
      }
      
      const { status } = await Audio.requestPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert('Permission Denied', 'Permission to access microphone is required!');
        return;
      }

      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
      });

      const recordingObject = new Audio.Recording();
      await recordingObject.prepareToRecordAsync(Audio.RECORDING_OPTIONS_PRESET_HIGH_QUALITY);
      await recordingObject.startAsync();

      setRecording(recordingObject);
    } catch (err) {
      console.error('Failed to start recording', err);
      Alert.alert('Recording Error', 'Failed to start voice recording');
    }
  };

  const stopRecording = async () => {
    if (!recording) return;

    try {
      await recording.stopAndUnloadAsync();
      const uri = recording.getURI();
      setRecording(null);
      handleVoiceInput(uri);
    } catch (error) {
      console.error('Error stopping recording:', error);
      Alert.alert('Processing Error', 'There was an error processing your voice input. Please try again.');
      setRecording(null);
    }
  };

  const clearChat = () => {
    if (isSpeaking) {
      stopSpeaking();
    }
    
    Alert.alert(
      'Clear Chat',
      'Are you sure you want to clear the chat history?',
      [
        {
          text: 'Cancel',
          style: 'cancel',
        },
        {
          text: 'Clear',
          onPress: () => {
            setMessages([
              {
                id: '1',
                text: "Hello! I'm your AushadX AI assistant. I can help you with medical questions, general health advice, or just have a friendly conversation. What's on your mind today?",
                sender: 'bot',
                timestamp: new Date(),
              },
            ]);
            setConversationContext([]);
          },
          style: 'destructive',
        },
      ]
    );
  };

  const formatTime = (timestamp) => {
    return timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar style="light" />
      <KeyboardAvoidingView 
        style={styles.container}
        behavior={Platform.OS === 'ios' ? 'padding' : 'padding'}
        keyboardVerticalOffset={Platform.OS === 'ios' ? 90 : 80}
      >
        <LinearGradient colors={['#121212', '#1E1E1E', '#252525']} style={styles.gradientBackground}>
          <View style={styles.header}>
            <TouchableOpacity onPress={navigation.goBack} style={styles.backButton}>
              <Ionicons name="arrow-back" size={24} color="white" />
            </TouchableOpacity>
            <View style={styles.headerInfo}>
              <Text style={styles.headerTitle}>AushadX AI Assistant</Text>
              <Text style={styles.headerSubtitle}>
                {isSpeaking ? 'Speaking...' : typingIndicator ? 'Typing...' : 'Online'}
              </Text>
            </View>
            <TouchableOpacity onPress={clearChat} style={styles.clearButton}>
              <Ionicons name="trash-outline" size={22} color="white" />
            </TouchableOpacity>
          </View>

          <FlatList
            ref={flatListRef}
            data={messages}
            keyExtractor={(item) => item.id}
            renderItem={({ item }) => (
              <View style={styles.messageContainer}>
                <TouchableOpacity 
                  activeOpacity={item.sender === 'bot' ? 0.7 : 1}
                  onPress={() => item.sender === 'bot' && speakMessage(item.text)}
                >
                  <View style={[styles.messageBubble, item.sender === 'user' ? styles.userMessage : styles.botMessage]}>
                    <Text style={styles.messageText}>{item.text}</Text>
                    <View style={styles.messageFooter}>
                      <Text style={styles.timestamp}>
                        {formatTime(item.timestamp)}
                      </Text>
                      {item.sender === 'bot' && (
                        <TouchableOpacity 
                          style={styles.speakButton}
                          onPress={() => speakMessage(item.text)}
                        >
                          <Ionicons name="volume-medium-outline" size={14} color="rgba(255,255,255,0.6)" />
                        </TouchableOpacity>
                      )}
                    </View>
                  </View>
                </TouchableOpacity>
              </View>
            )}
            contentContainerStyle={styles.messageList}
            showsVerticalScrollIndicator={false}
          />

          {typingIndicator && (
            <Animated.View style={[styles.typingIndicator, { opacity: typingAnimation }]}>
              <View style={styles.typingDots}>
                <View style={[styles.dot, styles.dot1]} />
                <View style={[styles.dot, styles.dot2]} />
                <View style={[styles.dot, styles.dot3]} />
              </View>
              <Text style={styles.typingText}>AushadX is thinking...</Text>
            </Animated.View>
          )}

          {isSpeaking && (
            <TouchableOpacity style={styles.stopSpeakingButton} onPress={stopSpeaking}>
              <Ionicons name="stop-circle" size={20} color="#FF4444" />
              <Text style={styles.stopSpeakingText}>Stop Speaking</Text>
            </TouchableOpacity>
          )}

          <View style={[styles.inputContainer,, { position: 'relative' }]}>
            <View style={styles.inputRow}>
              <TouchableOpacity
                onPress={recording ? stopRecording : startRecording}
                style={[styles.voiceButton, recording && styles.recordingButton]}
                disabled={isLoading}
              >
                <Ionicons name={recording ? 'stop-circle' : 'mic'} size={22} color="white" />
              </TouchableOpacity>
              <TextInput
                style={styles.input}
                value={inputText}
                onChangeText={setInputText}
                placeholder="Ask me anything about health or just chat..."
                placeholderTextColor="#9e9e9e"
                returnKeyType="send"
                onSubmitEditing={handleSendMessage}
                editable={!isLoading}
                multiline
                maxLength={500}
                textAlignVertical="center"
              />
              <TouchableOpacity
                style={[styles.sendButton, (!inputText.trim() || isLoading) && styles.disabledButton]}
                onPress={handleSendMessage}
                disabled={!inputText.trim() || isLoading}
                activeOpacity={0.7}
              >
                <Ionicons name="send" size={20} color="white" />
              </TouchableOpacity>
            </View>
          </View>
        </LinearGradient>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  safeArea: { 
    flex: 1, 
    backgroundColor: '#121212' 
  },
  container: {
    flex: 1,
  },
  gradientBackground: { 
    flex: 1, 
    width: '100%' 
  },
  header: { 
    flexDirection: 'row', 
    alignItems: 'center', 
    justifyContent: 'space-between', 
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(255,255,255,0.1)',
  },
  headerInfo: {
    flex: 1,
    alignItems: 'center',
  },
  headerTitle: { 
    fontSize: 18, 
    fontWeight: '600', 
    color: 'white' 
  },
  headerSubtitle: { 
    fontSize: 12, 
    color: 'rgba(255,255,255,0.6)', 
    marginTop: 2 
  },
  backButton: { 
    padding: 8 
  },
  clearButton: { 
    padding: 8 
  },
  messageList: { 
    paddingHorizontal: 16, 
    paddingBottom: 20, 
    paddingTop: 10,
    flexGrow: 1,
  },
  messageContainer: {
    marginBottom: 12,
  },
  messageBubble: { 
    maxWidth: '85%', 
    padding: 14, 
    borderRadius: 18,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
    elevation: 2,
  },
  userMessage: { 
    alignSelf: 'flex-end', 
    backgroundColor: '#4466EE',
    borderBottomRightRadius: 4,
  },
  botMessage: { 
    alignSelf: 'flex-start', 
    backgroundColor: '#2C2C2E',
    borderBottomLeftRadius: 4,
  },
  messageText: { 
    color: 'white', 
    fontSize: 16, 
    lineHeight: 22,
  },
  messageFooter: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: 8,
  },
  timestamp: {
    fontSize: 11,
    color: 'rgba(255,255,255,0.5)',
  },
  speakButton: {
    padding: 6,
    borderRadius: 12,
    backgroundColor: 'rgba(68,102,238,0.2)',
  },
  typingIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 12,
    paddingHorizontal: 16,
    backgroundColor: 'rgba(44,44,46,0.8)',
    borderRadius: 20,
    marginHorizontal: 16,
    marginBottom: 10,
  },
  typingDots: {
    flexDirection: 'row',
    marginRight: 8,
  },
  dot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: '#4466EE',
    marginHorizontal: 2,
  },
  typingText: {
    color: 'rgba(255,255,255,0.7)',
    fontSize: 14,
    fontStyle: 'italic',
  },
  stopSpeakingButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'rgba(255,68,68,0.15)',
    borderRadius: 20,
    paddingVertical: 8,
    paddingHorizontal: 16,
    marginHorizontal: 16,
    marginBottom: 10,
  },
  stopSpeakingText: {
    color: '#FF4444',
    fontSize: 14,
    fontWeight: '500',
    marginLeft: 6,
  },
  inputContainer: { 
    padding: 16,
    paddingBottom: Platform.OS === 'ios' ? 34 : 16,
    backgroundColor: 'rgba(18,18,18,0.95)',
    borderTopWidth: 1,
    borderTopColor: 'rgba(255,255,255,0.1)',
  },
  inputRow: { 
    flexDirection: 'row', 
    alignItems: 'flex-end', 
    backgroundColor: '#2C2C2E', 
    borderRadius: 24, 
    paddingHorizontal: 6,
    paddingVertical: 6,
    minHeight: 52,
    maxHeight: 120,
  },
  input: { 
    flex: 1, 
    color: 'white', 
    fontSize: 16, 
    paddingHorizontal: 16,
    paddingVertical: 12,
    maxHeight: 100,
    textAlignVertical: 'center',
  },
  voiceButton: { 
    backgroundColor: '#4466EE', 
    borderRadius: 20, 
    padding: 12, 
    marginRight: 8,
    alignItems: 'center',
    justifyContent: 'center',
  },
  recordingButton: { 
    backgroundColor: '#FF4444' 
  },
  sendButton: { 
    backgroundColor: '#4466EE', 
    borderRadius: 20, 
    padding: 12, 
    marginLeft: 8,
    alignItems: 'center',
    justifyContent: 'center',
  },
  disabledButton: { 
    backgroundColor: 'rgba(68,102,238,0.3)' 
  },
});

export default AIChat;