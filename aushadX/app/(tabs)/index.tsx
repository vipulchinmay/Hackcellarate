import { View, StyleSheet, Alert, ActivityIndicator, Text, Animated, Easing, Image, TouchableOpacity, SafeAreaView, StatusBar, Platform, Dimensions } from "react-native";
import { useState, useEffect, useRef } from "react";
import { useRouter } from "expo-router";
import ImageViewer from "@/components/ImageViewer";
import Button from "@/components/Button";
import * as ImagePicker from "expo-image-picker";
import { useNavigation } from '@react-navigation/native';
import { useLanguage } from "@/components/LanguageContext";
import translations from "@/components/translation";
import { BlurView } from "expo-blur";
import 'react-native-gesture-handler';
import { LinearGradient } from "expo-linear-gradient";
import { MaterialCommunityIcons, Ionicons, FontAwesome } from '@expo/vector-icons';
import LottieView from 'lottie-react-native';

const { width, height } = Dimensions.get('window');

export default function Index() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [abortController, setAbortController] = useState<AbortController | null>(null);
  const router = useRouter();
  const navigation = useNavigation();
  const { language } = useLanguage();
  const lottieRef = useRef(null);

  // Translation helper function
  const t = (key: string) => translations[language]?.[key] || key;

  // Enhanced animations
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const welcomeAnim = useRef(new Animated.Value(0)).current;
  const buttonScale = useRef(new Animated.Value(1)).current;
  const loadingRotation = useRef(new Animated.Value(0)).current;
  const buttonOpacity = useRef(new Animated.Value(1)).current;
  const imageSlideUp = useRef(new Animated.Value(50)).current;
  const imageScale = useRef(new Animated.Value(0.95)).current;
  const aiButtonAnim = useRef(new Animated.Value(0)).current;

  // Start rotation animation for loading spinner
  useEffect(() => {
    if (loading) {
      Animated.loop(
        Animated.timing(loadingRotation, {
          toValue: 1,
          duration: 1500,
          easing: Easing.linear,
          useNativeDriver: true
        })
      ).start();
      
      if (lottieRef.current) {
        lottieRef.current.play();
      }
    } else {
      loadingRotation.setValue(0);
      if (lottieRef.current) {
        lottieRef.current.pause();
      }
    }
  }, [loading]);

  // Enhanced welcome animation with sequence
  useEffect(() => {
    Animated.sequence([
      Animated.timing(welcomeAnim, {
        toValue: 1,
        duration: 800,
        easing: Easing.out(Easing.quad),
        useNativeDriver: true,
      }),
      Animated.timing(buttonOpacity, {
        toValue: 1,
        duration: 500,
        delay: 300,
        easing: Easing.out(Easing.quad),
        useNativeDriver: true,
      }),
      Animated.spring(aiButtonAnim, {
        toValue: 1,
        friction: 6,
        tension: 40,
        useNativeDriver: true,
      })
    ]).start();

    // Request camera permissions on component mount
    (async () => {
      const { status } = await ImagePicker.requestCameraPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert(t('Permission Needed'), t('Camera permission is needed to take photos'));
      }
    })();
  }, []);

  // Enhanced image selected animation
  useEffect(() => {
    if (selectedImage) {
      Animated.parallel([
        Animated.timing(fadeAnim, {
          toValue: 1,
          duration: 600,
          easing: Easing.out(Easing.cubic),
          useNativeDriver: true,
        }),
        Animated.timing(imageSlideUp, {
          toValue: 0,
          duration: 500,
          easing: Easing.out(Easing.quad),
          useNativeDriver: true,
        }),
        Animated.timing(imageScale, {
          toValue: 1,
          duration: 500,
          easing: Easing.out(Easing.back(1.5)),
          useNativeDriver: true,
        })
      ]).start();
    }
  }, [selectedImage]);

  // Enhanced button press animations
  const animateButtonPressIn = () => {
    Animated.parallel([
      Animated.spring(buttonScale, {
        toValue: 0.96,
        friction: 5,
        tension: 100,
        useNativeDriver: true,
      }),
      Animated.timing(buttonOpacity, {
        toValue: 0.9,
        duration: 100,
        useNativeDriver: true,
      })
    ]).start();
  };

  const animateButtonPressOut = () => {
    Animated.parallel([
      Animated.spring(buttonScale, {
        toValue: 1,
        friction: 4,
        tension: 100,
        useNativeDriver: true,
      }),
      Animated.timing(buttonOpacity, {
        toValue: 1,
        duration: 100,
        useNativeDriver: true,
      })
    ]).start();
  };

  const pickImage = async () => {
    const permissionResult = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permissionResult.granted) {
      Alert.alert(
        t("Permission Denied"),
        t("You need to grant gallery access to select a photo.")
      );
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      quality: 1,
    });

    if (!result.canceled) {
      setSelectedImage(result.assets[0].uri);
    }
  };

  const takePhoto = async () => {
    try {
      const { status } = await ImagePicker.requestCameraPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert(
          t("Permission Denied"),
          t("You need to grant camera access to take a photo.")
        );
        return;
      }

      const result = await ImagePicker.launchCameraAsync({
        allowsEditing: true,
        aspect: [4, 3],
        quality: 0.8,
      });
      
      if (!result.canceled) {
        setSelectedImage(result.assets[0].uri);
      }
    } catch (error) {
      console.error("Error taking picture:", error);
      Alert.alert(t("Error"), t("Failed to take photo. Please try again."));
    }
  };

  const abortProcessing = () => {
    if (abortController) {
      abortController.abort();
      setLoading(false);
      setAbortController(null);
      
      // iOS-style alert with haptic feedback (if available)
      if (Platform.OS === 'ios') {
        // iOS would typically use haptic feedback here
      }
      Alert.alert(t("Processing Aborted"), t("Image processing has been stopped."));
    }
  };

  const scanImage = async () => {
    if (!selectedImage) {
      Alert.alert(t("No Image Selected"), t("Please choose or take a photo first."));
      return;
    }

    const controller = new AbortController();
    setAbortController(controller);
    setLoading(true);

    try {
      const response = await fetch(selectedImage, { signal: controller.signal });
      const blob = await response.blob();
      const reader = new FileReader();
      reader.readAsDataURL(blob);

      reader.onloadend = async () => {
        const base64Image = reader.result.split(",")[1];

        // Send the selected language along with the image
        const serverResponse = await fetch("http://192.168.1.102:5000/scan", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ 
            image: base64Image,
            language: language
          }),
          signal: controller.signal,
        });

        const data = await serverResponse.json();
        setLoading(false);

        if (data.error) {
          Alert.alert(t("Error"), t("Failed to extract medicine details."));
          return;
        }

        navigation.navigate("Schedule", {
          raw_response: data.raw_response,
        });
      };
    } catch (error) {
      if (error.name === "AbortError") {
        console.log("Image processing was aborted.");
      } else {
        console.error("Error scanning image:", error);
        Alert.alert(t("Error"), t("Something went wrong while scanning."));
      }
      setLoading(false);
    }
  };

  // Navigate to AI chat screen
  const openAIChat = () => {
    // Animate button press
    Animated.sequence([
      Animated.timing(buttonScale, {
        toValue: 0.9,
        duration: 100,
        useNativeDriver: true
      }),
      Animated.timing(buttonScale, {
        toValue: 1,
        duration: 100,
        useNativeDriver: true
      })
    ]).start();
    
    // Navigate to the chat screen
    navigation.navigate("AIChat");
  };

  // Animation interpolations
  const spinInterpolation = loadingRotation.interpolate({
    inputRange: [0, 1],
    outputRange: ['0deg', '360deg']
  });

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar barStyle="light-content" />
      <LinearGradient
        colors={['#121212', '#1E1E1E', '#252525']}
        style={styles.gradientBackground}
      >
        <View style={styles.container}>
          {/* AI Chat Button */}
          <Animated.View 
            style={[
              styles.aiButtonContainer,
              {
                opacity: aiButtonAnim,
                transform: [
                  { scale: aiButtonAnim },
                  { translateY: aiButtonAnim.interpolate({
                    inputRange: [0, 1],
                    outputRange: [10, 0]
                  })}
                ]
              }
            ]}
          >
            <TouchableOpacity
              style={styles.aiButton}
              onPress={openAIChat}
              activeOpacity={0.8}
            >
              <LinearGradient
                colors={['#4466EE', '#5D5DFD']}
                style={styles.aiButtonGradient}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 1 }}
              >
                <FontAwesome name="comment-medical" size={22} color="white" />
              </LinearGradient>
            </TouchableOpacity>
          </Animated.View>

          {!selectedImage ? (
            <Animated.View 
              style={[
                styles.welcomeContainer, 
                { 
                  opacity: welcomeAnim,
                  transform: [{ translateY: welcomeAnim.interpolate({
                    inputRange: [0, 1],
                    outputRange: [20, 0]
                  })}]
                }
              ]}
            >
              <Image 
                source={require('@/assets/images/icon.png')} 
                style={styles.logo} 
                resizeMode="contain"
              />
              <Text style={styles.appTitle}>{t("AushadX")}</Text>
              <Text style={styles.welcomeText}>
                {t("🙏Welcome to AushadX! Tap 'Choose a photo'📸 to get started✨.")}
              </Text>
              
              {/* Add an illustration or animation for empty state */}
              <View style={styles.illustrationContainer}>
                <Image 
                  source={require('@/assets/images/scan-illustration.gif')} 
                  style={styles.illustration} 
                  resizeMode="contain"
                />
              </View>
            </Animated.View>
          ) : (
            <Animated.View 
              style={[
                styles.imageContainer, 
                { 
                  opacity: fadeAnim,
                  transform: [
                    { translateY: imageSlideUp },
                    { scale: imageScale }
                  ] 
                }
              ]}
            >
              <ImageViewer imgSource={selectedImage} />
              {selectedImage && (
                <TouchableOpacity 
                  style={styles.resetButton}
                  onPress={() => setSelectedImage(null)}
                >
                  <Ionicons name="close-circle" size={32} color="white" />
                </TouchableOpacity>
              )}
            </Animated.View>
          )}

          <Animated.View 
            style={[
              styles.footerContainer, 
              { 
                opacity: buttonOpacity,
                transform: [{ scale: buttonScale }]
              }
            ]}
          >
            <BlurView intensity={40} tint="dark" style={styles.blurOverlay}>
              {/* Show prominent action buttons when no image selected */}
              {!selectedImage ? (
                <View style={styles.mainActionButtonsContainer}>
                  <TouchableOpacity 
                    style={styles.mainActionButton}
                    onPress={takePhoto}
                    activeOpacity={0.9}
                  >
                    <View style={styles.buttonIconContainer}>
                      <Ionicons name="camera" size={28} color="#ffffff" />
                    </View>
                    <Text style={styles.mainActionButtonText}>{t("Take Photo")}</Text>
                  </TouchableOpacity>
                  
                  <View style={styles.buttonDivider} />
                  
                  <TouchableOpacity 
                    style={styles.mainActionButton}
                    onPress={pickImage}
                    activeOpacity={0.9}
                  >
                    <View style={styles.buttonIconContainer}>
                      <Ionicons name="images" size={28} color="#ffffff" />
                    </View>
                    <Text style={styles.mainActionButtonText}>{t("Gallery")}</Text>
                  </TouchableOpacity>
                </View>
              ) : (
                // Show Analyze button prominently when image is selected
                <View style={styles.imageSelectedActionContainer}>
                  {loading ? (
                    <View style={styles.loadingContainer}>
                      {/* Replace with Lottie animation if available */}
                      <View style={styles.lottieContainer}>
                        {/* <LottieView
                          ref={lottieRef}
                          source={require('@/assets/animations/scanning-animation.json')}
                          style={styles.lottie}
                          autoPlay
                          loop
                        /> */}
                        <Animated.View style={{ transform: [{ rotate: spinInterpolation }] }}>
                          <MaterialCommunityIcons name="loading" size={24} color="white" />
                        </Animated.View>
                      </View>
                      <Text style={styles.loadingText}>{t("Analyzing Medicine...")}</Text>
                      
                      <TouchableOpacity 
                        style={styles.floatingAbortButton} 
                        onPress={abortProcessing}
                        activeOpacity={0.9}
                      >
                        <Ionicons name="close-circle" size={18} color="white" />
                        <Text style={styles.abortButtonText}>{t("Cancel")}</Text>
                      </TouchableOpacity>
                    </View>
                  ) : (
                    <View style={styles.analyzeActionsContainer}>
                      <TouchableOpacity
                        style={styles.analyzeButtonEnhanced}
                        onPress={scanImage}
                        disabled={loading}
                        activeOpacity={0.85}
                      >
                        <MaterialCommunityIcons name="text-recognition" size={24} color="white" />
                        <Text style={styles.analyzeButtonTextEnhanced}>{t("Analyze Medicine")}</Text>
                      </TouchableOpacity>
                      
                      <View style={styles.secondaryActionsRow}>
                        <TouchableOpacity 
                          style={styles.secondaryButton}
                          onPress={takePhoto}
                          disabled={loading}
                        >
                          <Ionicons name="camera-outline" size={20} color="#ffffff" />
                          <Text style={styles.secondaryButtonText}>{t("Retake")}</Text>
                        </TouchableOpacity>
                        
                        <TouchableOpacity 
                          style={styles.secondaryButton}
                          onPress={pickImage}
                          disabled={loading}
                        >
                          <Ionicons name="images-outline" size={20} color="#ffffff" />
                          <Text style={styles.secondaryButtonText}>{t("Gallery")}</Text>
                        </TouchableOpacity>
                      </View>
                    </View>
                  )}
                </View>
              )}
            </BlurView>
          </Animated.View>
        </View>
      </LinearGradient>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: "#121212",
  },
  gradientBackground: {
    flex: 1,
    width: '100%',
  },
  container: {
    flex: 1,
    alignItems: "center",
    justifyContent: "space-between",
    paddingTop: Platform.OS === 'android' ? StatusBar.currentHeight : 0,
    paddingBottom: 0,
  },
  // New AI button styles
  aiButtonContainer: {
    position: 'absolute',
    top: Platform.OS === 'ios' ? 12 : StatusBar.currentHeight + 12 || 12,
    right: 16,
    zIndex: 100,
  },
  aiButton: {
    width: 56,
    height: 56,
    borderRadius: 28,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 6,
    elevation: 8,
  },
  aiButtonGradient: {
    width: 56,
    height: 56,
    borderRadius: 28,
    justifyContent: 'center',
    alignItems: 'center',
  },
  welcomeContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    width: "100%",
    paddingHorizontal: 20,
    paddingTop: 40,
  },
  logo: {
    width: 90,
    height: 90,
    marginBottom: 16,
  },
  appTitle: {
    fontSize: 36,
    fontWeight: "bold",
    color: "#ffffff",
    marginBottom: 12,
    letterSpacing: 0.5,
  },
  welcomeText: {
    fontSize: 18,
    color: "#e0e0e0",
    textAlign: "center",
    lineHeight: 24,
    maxWidth: "80%",
    marginBottom: 30,
  },
  illustrationContainer: {
    width: width * 0.7,
    height: width * 0.5,
    marginTop: 20,
    marginBottom: 30,
    justifyContent: 'center',
    alignItems: 'center',
  },
  illustration: {
    width: '100%',
    height: '100%',
    opacity: 0.9,
  },
  imageContainer: {
    width: width * 0.9,
    height: height * 0.55,
    marginTop: 40,
    marginBottom: 20,
    borderRadius: 24,
    overflow: "hidden",
    backgroundColor: "#2c2c2c",
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.4,
    shadowRadius: 12,
    elevation: 10,
  },
  resetButton: {
    position: "absolute",
    top: 12,
    right: 12,
    backgroundColor: "rgba(0, 0, 0, 0.6)",
    borderRadius: 20,
    padding: 4,
    zIndex: 10,
  },
  footerContainer: {
    width: "100%",
    paddingHorizontal: 0,
    marginTop: 'auto',
  },
  blurOverlay: {
    borderTopLeftRadius: 28,
    borderTopRightRadius: 28,
    overflow: "hidden",
    padding: 20,
    paddingBottom: Platform.OS === 'ios' ? 36 : 24,
    backgroundColor: "rgba(35, 35, 35, 0.8)",
  },
  // New styles for more intuitive UI
  mainActionButtonsContainer: {
    flexDirection: 'row',
    width: '100%',
    justifyContent: 'space-around',
    alignItems: 'center',
    height: 100,
  },
  mainActionButton: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    height: '100%',
  },
  buttonDivider: {
    width: 1,
    height: '50%',
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
  },
  buttonIconContainer: {
    width: 64,
    height: 64,
    borderRadius: 32,
    backgroundColor: 'rgba(255, 255, 255, 0.15)',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 8,
  },
  mainActionButtonText: {
    color: '#ffffff',
    fontWeight: '600',
    fontSize: 16,
  },
  
  // Image selected content
  imageSelectedActionContainer: {
    width: '100%',
  },
  loadingContainer: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 10,
  },
  lottieContainer: {
    width: 70,
    height: 70,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 10,
  },
  lottie: {
    width: '100%',
    height: '100%',
  },
  loadingText: {
    color: "white",
    fontWeight: "600",
    fontSize: 18,
    marginBottom: 10,
  },
  floatingAbortButton: {
    backgroundColor: "rgba(255, 59, 48, 0.9)",
    borderRadius: 20,
    paddingVertical: 8,
    paddingHorizontal: 16,
    marginTop: 16,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 6,
  },
  abortButtonText: {
    color: "white",
    fontWeight: "600",
    fontSize: 15,
  },
  
  analyzeActionsContainer: {
    width: '100%',
  },
  analyzeButtonEnhanced: {
    backgroundColor: "#34C759", // iOS green
    borderRadius: 16,
    paddingVertical: 16,
    paddingHorizontal: 20,
    alignItems: "center",
    justifyContent: "center",
    width: "100%",
    flexDirection: "row",
    gap: 10,
    shadowColor: "#34C759",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 5,
    marginBottom: 16,
  },
  analyzeButtonTextEnhanced: {
    color: "#ffffff",
    fontWeight: "bold",
    fontSize: 18,
  },
  secondaryActionsRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    width: '100%',
    gap: 12,
  },
  secondaryButton: {
    flex: 1,
    backgroundColor: "rgba(255, 255, 255, 0.15)",
    borderRadius: 14,
    paddingVertical: 12,
    paddingHorizontal: 16,
    alignItems: "center",
    justifyContent: "center",
    flexDirection: "row",
    gap: 8,
  },
  secondaryButtonText: {
    color: "#ffffff",
    fontWeight: "500",
    fontSize: 15,
  },
  
  // Legacy styles for compatibility
  buttonGroup: {
    width: "100%",
    flexDirection: "row",
    justifyContent: "space-between",
    gap: 12,
    marginBottom: 16,
  },
  buttonContainer: {
    flex: 1,
  },
  cameraButton: {
    backgroundColor: "#007AFF", // iOS blue
    borderRadius: 16,
    paddingVertical: 14,
    paddingHorizontal: 20,
    alignItems: "center",
    justifyContent: "center",
    flexDirection: "row",
    gap: 8,
    shadowColor: "#007AFF",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.3,
    shadowRadius: 4,
    elevation: 3,
  },
  galleryButton: {
    backgroundColor: "#5856D6", // iOS purple
    borderRadius: 16,
    paddingVertical: 14,
    paddingHorizontal: 20,
    alignItems: "center",
    justifyContent: "center",
    flexDirection: "row",
    gap: 8,
    shadowColor: "#5856D6",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.3,
    shadowRadius: 4,
    elevation: 3,
  },
  buttonText: {
    color: "#ffffff",
    fontWeight: "600",
    fontSize: 16,
  },
  actionButtonsContainer: {
    width: "100%",
    alignItems: "center",
    marginTop: 8,
  },
  analyzeButton: {
    backgroundColor: "#34C759", // iOS green
    borderRadius: 16,
    paddingVertical: 14,
    paddingHorizontal: 24,
    alignItems: "center",
    justifyContent: "center",
    width: "100%",
    flexDirection: "row",
    gap: 8,
    shadowColor: "#34C759",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.3,
    shadowRadius: 4,
    elevation: 3,
  },
  analyzeButtonDisabled: {
    backgroundColor: "#34C75980", // iOS green with opacity
  },
  analyzeButtonText: {
    color: "#ffffff",
    fontWeight: "bold",
    fontSize: 17,
  },
  abortButton: {
    backgroundColor: "#FF3B30", // iOS red
    borderRadius: 16,
    paddingVertical: 12,
    paddingHorizontal: 20,
    marginTop: 12,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 6,
    shadowColor: "#FF3B30",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.3,
    shadowRadius: 4,
    elevation: 3,
    width: "50%",
  },
});