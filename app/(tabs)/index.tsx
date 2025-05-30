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
  const headerAnim = useRef(new Animated.Value(0)).current;

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
        }),
        Animated.timing(headerAnim, {
          toValue: 1,
          duration: 400,
          delay: 200,
          easing: Easing.out(Easing.quad),
          useNativeDriver: true,
        })
      ]).start();
    } else {
      // Reset animations when no image
      fadeAnim.setValue(0);
      imageSlideUp.setValue(50);
      imageScale.setValue(0.95);
      headerAnim.setValue(0);
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
      aspect: [4, 3],
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
        const serverResponse = await fetch("http://192.168.1.177:5000/scan", {
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

  //Navigate to AI chat screen
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
          {/* <Animated.View 
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
                colors={['#f84040', '#5D5DFD']}
                style={styles.aiButtonGradient}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 1 }}
              >
                <Ionicons name="fitness-outline" size={40} color="white" />
              </LinearGradient>
            </TouchableOpacity>
          </Animated.View> */}

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
            <View style={styles.imageSelectedContainer}>
              {/* Enhanced Header */}
              <Animated.View 
                style={[
                  styles.imageHeader,
                  {
                    opacity: headerAnim,
                    transform: [{ translateY: headerAnim.interpolate({
                      inputRange: [0, 1],
                      outputRange: [-20, 0]
                    })}]
                  }
                ]}
              >
                <View style={styles.headerContent}>
                  <View style={styles.headerTextContainer}>
                    <Text style={styles.headerTitle}>{t("Medicine Image")}</Text>
                    <Text style={styles.headerSubtitle}>
                      {loading ? t("Analyzing...") : t("Ready to analyze")}
                    </Text>
                  </View>
                  <TouchableOpacity 
                    style={styles.changeImageButton}
                    onPress={() => setSelectedImage(null)}
                    disabled={loading}
                  >
                    <Ionicons name="refresh-outline" size={20} color="#007AFF" />
                    <Text style={styles.changeImageText}>{t("Change")}</Text>
                  </TouchableOpacity>
                </View>
              </Animated.View>

              {/* Enhanced Image Container with proper aspect ratio */}
              <Animated.View 
                style={[
                  styles.enhancedImageContainer, 
                  { 
                    opacity: fadeAnim,
                    transform: [
                      { translateY: imageSlideUp },
                      { scale: imageScale }
                    ] 
                  }
                ]}
              >
                {/* Image with proper aspect ratio handling */}
                <View style={styles.imageWrapper}>
                  <Image 
                    source={{ uri: selectedImage }}
                    style={styles.selectedImage}
                    resizeMode="cover"
                  />
                  
                  {/* Stylish border overlay */}
                  <View style={styles.imageBorder} />
                  
                  {/* Loading overlay */}
                  {loading && (
                    <View style={styles.loadingOverlay}>
                      <BlurView intensity={20} tint="dark" style={styles.loadingBlur}>
                        <View style={styles.loadingContent}>
                          <Animated.View style={{ transform: [{ rotate: spinInterpolation }] }}>
                            <MaterialCommunityIcons name="loading" size={32} color="white" />
                          </Animated.View>
                          <Text style={styles.overlayLoadingText}>{t("Scanning...")}</Text>
                          
                          {/* Progress dots */}
                          <View style={styles.progressDots}>
                            {[0, 1, 2].map(i => (
                              <Animated.View 
                                key={i}
                                style={[
                                  styles.progressDot,
                                  {
                                    opacity: loadingRotation.interpolate({
                                      inputRange: [0, 0.33, 0.66, 1],
                                      outputRange: i === 0 ? [1, 0.3, 0.3, 1] : 
                                                 i === 1 ? [0.3, 1, 0.3, 0.3] : 
                                                          [0.3, 0.3, 1, 0.3]
                                    })
                                  }
                                ]}
                              />
                            ))}
                          </View>
                        </View>
                      </BlurView>
                    </View>
                  )}
                  
                  {/* Image quality indicator */}
                  <View style={styles.qualityIndicator}>
                    <View style={styles.qualityDot} />
                    <Text style={styles.qualityText}>{t("High Quality")}</Text>
                  </View>
                  
                  {/* Corner accents */}
                  <View style={styles.cornerAccents}>
                    <View style={[styles.cornerAccent, styles.topLeft]} />
                    <View style={[styles.cornerAccent, styles.topRight]} />
                    <View style={[styles.cornerAccent, styles.bottomLeft]} />
                    <View style={[styles.cornerAccent, styles.bottomRight]} />
                  </View>
                </View>
                
                {/* Enhanced Image info bar */}
                <View style={styles.imageInfoBar}>
                  <View style={styles.imageInfoItem}>
                    <Ionicons name="checkmark-circle" size={16} color="#34C759" />
                    <Text style={styles.imageInfoText}>{t("Image Ready")}</Text>
                  </View>
                  <View style={styles.imageInfoItem}>
                    <Ionicons name="scan" size={16} color="#007AFF" />
                    <Text style={styles.imageInfoText}>{t("OCR Ready")}</Text>
                  </View>
                  <View style={styles.imageInfoItem}>
                    <MaterialCommunityIcons name="pill" size={16} color="#FF9500" />
                    <Text style={styles.imageInfoText}>{t("Medicine")}</Text>
                  </View>
                </View>
              </Animated.View>
            </View>
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
                // Enhanced UI when image is selected
                <View style={styles.imageSelectedActionContainer}>
                  {loading ? (
                    <View style={styles.loadingContainer}>
                      <View style={styles.loadingHeader}>
                        <Text style={styles.loadingTitle}>{t("Analyzing Medicine")}</Text>
                        <Text style={styles.loadingSubtitle}>{t("Please wait while we process your image...")}</Text>
                      </View>
                      
                      <View style={styles.loadingProgressContainer}>
                        <Animated.View style={{ transform: [{ rotate: spinInterpolation }] }}>
                          <MaterialCommunityIcons name="loading" size={28} color="#007AFF" />
                        </Animated.View>
                      </View>
                      
                      <TouchableOpacity 
                        style={styles.modernAbortButton} 
                        onPress={abortProcessing}
                        activeOpacity={0.9}
                      >
                        <Ionicons name="stop-circle-outline" size={20} color="#FF3B30" />
                        <Text style={styles.modernAbortText}>{t("Cancel")}</Text>
                      </TouchableOpacity>
                    </View>
                  ) : (
                    <View style={styles.modernActionsContainer}>
                      {/* Primary action */}
                      <TouchableOpacity
                        style={styles.primaryAnalyzeButton}
                        onPress={scanImage}
                        disabled={loading}
                        activeOpacity={0.85}
                      >
                        <LinearGradient
                          colors={['#34C759', '#28A745']}
                          style={styles.primaryButtonGradient}
                          start={{ x: 0, y: 0 }}
                          end={{ x: 1, y: 1 }}
                        >
                          <MaterialCommunityIcons name="text-recognition" size={24} color="white" />
                          <Text style={styles.primaryButtonText}>{t("Analyze Medicine")}</Text>
                          <Ionicons name="arrow-forward" size={20} color="white" />
                        </LinearGradient>
                      </TouchableOpacity>
                      
                      {/* Secondary actions */}
                      <View style={styles.secondaryActionsContainer}>
                        <TouchableOpacity 
                          style={styles.modernSecondaryButton}
                          onPress={takePhoto}
                          disabled={loading}
                        >
                          <View style={styles.secondaryButtonContent}>
                            <Ionicons name="camera-outline" size={20} color="#007AFF" />
                            <Text style={styles.modernSecondaryText}>{t("Retake")}</Text>
                          </View>
                        </TouchableOpacity>
                        
                        <TouchableOpacity 
                          style={styles.modernSecondaryButton}
                          onPress={pickImage}
                          disabled={loading}
                        >
                          <View style={styles.secondaryButtonContent}>
                            <Ionicons name="images-outline" size={20} color="#007AFF" />
                            <Text style={styles.modernSecondaryText}>{t("Gallery")}</Text>
                          </View>
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
  // AI button styles
  // aiButtonContainer: {
  //   position: 'absolute',
  //   top: Platform.OS === 'ios' ? 12 : StatusBar.currentHeight + 12 || 12,
  //   right: 16,
  //   zIndex: 100,
  // },
  // aiButton: {
  //   width: 56,
  //   height: 56,
  //   borderRadius: 28,
  //   shadowColor: "#000",
  //   shadowOffset: { width: 0, height: 4 },
  //   shadowOpacity: 0.3,
  //   shadowRadius: 6,
  //   elevation: 8,
  // },
  // aiButtonGradient: {
  //   width: 56,
  //   height: 56,
  //   borderRadius: 28,
  //   justifyContent: 'center',
  //   alignItems: 'center',
  // },
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
  
  // Enhanced image selected container
  imageSelectedContainer: {
    flex: 1,
    width: '100%',
    paddingTop: 20,
  },
  
  // Enhanced header
  imageHeader: {
    paddingHorizontal: 20,
    marginBottom: 16,
  },
  headerContent: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  headerTextContainer: {
    flex: 1,
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#ffffff',
    marginBottom: 4,
  },
  headerSubtitle: {
    fontSize: 14,
    color: '#a0a0a0',
  },
  changeImageButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(0, 122, 255, 0.15)',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 12,
    gap: 4,
  },
  changeImageText: {
    color: '#007AFF',
    fontSize: 14,
    fontWeight: '600',
  },
  
  // Enhanced image container
  enhancedImageContainer: {
    flex: 1,
    marginHorizontal: 16,
    marginBottom: 20,
    borderRadius: 24,
    backgroundColor: '#1a1a1a',
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 12 },
    shadowOpacity: 0.6,
    shadowRadius: 16,
    elevation: 12,
    overflow: 'hidden',
  },
  imageWrapper: {
    flex: 1,
    position: 'relative',
    aspectRatio: 4/3,
    minHeight: height * 0.4,
  },
  selectedImage: {
    width: '100%',
    height: '100%',
    borderTopLeftRadius: 24,
    borderTopRightRadius: 24,
  },
  imageBorder: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    borderTopLeftRadius: 24,
    borderTopRightRadius: 24,
    borderWidth: 2,
    borderColor: 'rgba(255, 255, 255, 0.1)',
  },
  loadingOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    justifyContent: 'center',
    alignItems: 'center',
    borderTopLeftRadius: 24,
    borderTopRightRadius: 24,
  },
  loadingBlur: {
    width: '100%',
    height: '100%',
    justifyContent: 'center',
    alignItems: 'center',
    borderTopLeftRadius: 24,
    borderTopRightRadius: 24,
  },
  loadingContent: {
    alignItems: 'center',
    gap: 16,
  },
  overlayLoadingText: {
    color: 'white',
    fontSize: 18,
    fontWeight: '600',
  },
  progressDots: {
    flexDirection: 'row',
    gap: 8,
    marginTop: 8,
  },
  progressDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: 'white',
  },
  qualityIndicator: {
    position: 'absolute',
    top: 16,
    left: 16,
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 12,
    gap: 6,
  },
  qualityDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: '#34C759',
  },
  qualityText: {
    color: 'white',
    fontSize: 12,
    fontWeight: '600',
  },
  cornerAccents: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
  },
  cornerAccent: {
    position: 'absolute',
    width: 20,
    height: 20,
    borderColor: '#007AFF',
    borderWidth: 3,
  },
  topLeft: {
    top: 12,
    left: 12,
    borderRightWidth: 0,
    borderBottomWidth: 0,
    borderTopLeftRadius: 8,
  },
  topRight: {
    top: 12,
    right: 12,
    borderLeftWidth: 0,
    borderBottomWidth: 0,
    borderTopRightRadius: 8,
  },
  bottomLeft: {
    bottom: 12,
    left: 12,
    borderRightWidth: 0,
    borderTopWidth: 0,
    borderBottomLeftRadius: 8,
  },
  bottomRight: {
    bottom: 12,
    right: 12,
    borderLeftWidth: 0,
    borderTopWidth: 0,
    borderBottomRightRadius: 8,
  },
  imageInfoBar: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    alignItems: 'center',
    backgroundColor: '#1a1a1a',
    paddingVertical: 12,
    paddingHorizontal: 16,
    borderBottomLeftRadius: 24,
    borderBottomRightRadius: 24,
    borderTopWidth: 1,
    borderTopColor: 'rgba(255, 255, 255, 0.1)',
  },
  imageInfoItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  imageInfoText: {
    color: '#a0a0a0',
    fontSize: 12,
    fontWeight: '500',
  },

  // Footer container
  footerContainer: {
    width: '100%',
    paddingBottom: Platform.OS === 'ios' ? 34 : 20,
  },
  blurOverlay: {
    borderTopLeftRadius: 24,
    borderTopRightRadius: 24,
    overflow: 'hidden',
    marginHorizontal: 16,
  },

  // Main action buttons (when no image selected)
  mainActionButtonsContainer: {
    flexDirection: 'row',
    paddingVertical: 20,
    paddingHorizontal: 24,
  },
  mainActionButton: {
    flex: 1,
    alignItems: 'center',
    paddingVertical: 20,
    paddingHorizontal: 16,
  },
  buttonIconContainer: {
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: 'rgba(0, 122, 255, 0.2)',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 12,
  },
  mainActionButtonText: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: '600',
  },
  buttonDivider: {
    width: 1,
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    marginVertical: 20,
  },

  // Image selected actions
  imageSelectedActionContainer: {
    paddingVertical: 16,
    paddingHorizontal: 20,
  },

  // Loading container
  loadingContainer: {
    alignItems: 'center',
    paddingVertical: 20,
  },
  loadingHeader: {
    alignItems: 'center',
    marginBottom: 20,
  },
  loadingTitle: {
    color: '#ffffff',
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 4,
  },
  loadingSubtitle: {
    color: '#a0a0a0',
    fontSize: 14,
    textAlign: 'center',
  },
  loadingProgressContainer: {
    marginBottom: 20,
  },
  modernAbortButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(255, 59, 48, 0.15)',
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: 12,
    gap: 8,
  },
  modernAbortText: {
    color: '#FF3B30',
    fontSize: 16,
    fontWeight: '600',
  },

  // Modern actions container
  modernActionsContainer: {
    gap: 16,
  },
  primaryAnalyzeButton: {
    borderRadius: 16,
    shadowColor: "#34C759",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 8,
  },
  primaryButtonGradient: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 18,
    paddingHorizontal: 24,
    borderRadius: 16,
    gap: 12,
  },
  primaryButtonText: {
    color: 'white',
    fontSize: 18,
    fontWeight: '700',
    flex: 1,
    textAlign: 'center',
  },

  // Secondary actions
  secondaryActionsContainer: {
    flexDirection: 'row',
    gap: 12,
  },
  modernSecondaryButton: {
    flex: 1,
    backgroundColor: 'rgba(0, 122, 255, 0.1)',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: 'rgba(0, 122, 255, 0.3)',
  },
  secondaryButtonContent: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 14,
    paddingHorizontal: 16,
    gap: 8,
  },
  modernSecondaryText: {
    color: '#007AFF',
    fontSize: 16,
    fontWeight: '600',
  },
});