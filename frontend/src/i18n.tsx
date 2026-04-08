import React, { createContext, useContext, useEffect, useMemo, useState } from "react";

export type Locale = "en" | "es" | "fr";

type Messages = Record<string, string>;

const STORAGE_KEY = "mpp:locale";

const translations: Record<Locale, Messages> = {
  en: {
    "nav.home": "Home",
    "nav.analyze": "Analyze",
    "nav.optimize": "Optimize",
    "nav.how": "How It Works",
    "nav.help": "Help",
    "nav.results": "Results",
    "app.startAnalyzing": "Start Analyzing",
    "app.resumeResult": "Resume latest result",
    "app.locale": "Language",
    "home.heroEyebrow": "The Digital Maestro",
    "home.heroTitle": "Predict a song's streaming potential before release.",
    "home.heroBody": "A premium academic prototype for producers, artists, and A&R teams. Analyze audio, score popularity from musical features, and explore what to change next.",
    "home.ctaUpload": "Analyze an MP3",
    "home.ctaSearch": "Search by Spotify ID",
    "home.ctaOptimize": "Open Optimizer",
    "home.metricRoutes": "3 core flows",
    "home.metricPrediction": "0-100 score",
    "home.metricTrust": "R2 about 0.39",
    "home.metricRoutesBody": "Upload, Spotify ID lookup, and parameter optimization.",
    "home.metricPredictionBody": "Machine learning projection based on extracted music features.",
    "home.metricTrustBody": "Designed as a decision support layer, not a guarantee of success.",
    "home.preview": "Signal Preview",
    "home.currentPosture": "Current posture",
    "home.estimatedScore": "Estimated popularity score",
    "home.routeUnreleased": "Best route for unreleased tracks",
    "home.routeReferences": "Best route for references",
    "home.uploadRouteTitle": "Analyze a local master file",
    "home.uploadRouteBody": "Upload an audio file, extract the feature set, and run a popularity prediction with a studio-like analysis surface.",
    "home.searchRouteTitle": "Use a Spotify track ID",
    "home.searchRouteBody": "Retrieve a track profile and predict popularity without uploading a file.",
    "home.optimizeTitle": "Fine-tune audio parameters with controlled locks",
    "home.optimizeBody": "Start from a baseline feature profile, lock what should remain stable, and let the model search for stronger combinations.",
    "home.openAudio": "Open audio analysis",
    "home.openLookup": "Open track lookup",
    "home.ua1": "Data-Driven Producer",
    "home.ua1Body": "Upload a master file, understand the score drivers, and test what to adjust before release.",
    "home.ua2": "Independent Artist",
    "home.ua2Body": "Check streaming potential with a Spotify track ID and get clear guidance without data science jargon.",
    "home.ua3": "A&R Analyst",
    "home.ua3Body": "Use a repeatable prediction workflow to compare candidate songs with a shared, objective signal.",
    "analyze.title": "Analyze a song",
    "analyze.body": "Choose the route that fits your source material. Both flows end in a shared results screen and can continue into optimization.",
    "analyze.uploadTab": "Upload Audio",
    "analyze.searchTab": "Spotify ID Search",
    "how.title": "How it works",
    "how.body": "The product flow follows the documented journeys: select a route, generate a feature set, receive a prediction, then iterate with optimization.",
    "how.step1": "Choose an analysis route",
    "how.step1Body": "Start from an uploaded audio file or from a Spotify track ID, depending on what you have available.",
    "how.step2": "Generate a popularity signal",
    "how.step2Body": "The model extracts features and predicts a 0-100 popularity score from the song profile.",
    "how.step3": "Interpret and improve",
    "how.step3Body": "Use the results as a support tool, then iterate with the optimization wizard to test better configurations.",
    "help.title": "Help and limitations",
    "help.body": "This product is a decision-support tool. It estimates popularity from audio features and leaves out external factors that strongly influence real-world performance.",
    "help.guidanceTitle": "Interpreting the score responsibly",
    "help.guidanceBody": "Use the prediction to support release conversations, not to replace artistic judgment. When one route fails, try the alternative source route.",
    "help.limit1": "Predictions estimate popularity from audio features only.",
    "help.limit2": "Marketing, artist reach, playlist placement, and release timing are not included.",
    "help.limit3": "The academic prototype is public and designed for single-song analysis, not batch processing.",
    "help.limit4": "Minority genres may produce less reliable scores than the top genres represented in the training data.",
    "results.title": "Analysis results",
    "results.body": "This result is the current session reference. You can continue into optimization without reprocessing the song.",
    "results.uploadRoute": "Upload route",
    "results.searchRoute": "Spotify route",
    "results.genre": "Genre",
    "results.score": "Popularity score",
    "results.tempo": "Tempo",
    "results.energy": "Energy",
    "results.duration": "Duration",
    "results.limitTitle": "Model limitation notice",
    "results.limitBody": "This score is an estimate based on audio features only. R2 is about 0.39, so factors such as marketing, artist exposure, playlist placement, and release timing are not included.",
    "results.analyzeAnother": "Analyze another song",
    "results.whatDoesMean": "What does this score mean?",
    "results.optimize": "Optimize this song",
    "results.viewFeatures": "View processed feature set",
    "results.supportive": "Supportive signal",
    "results.watch": "Constraint to review",
    "results.context": "Context signal",
    "optimize.title": "Optimize parameters",
    "optimize.body": "Keep the features that define the song, free the rest, and let the backend search for a stronger score.",
    "optimize.emptyTitle": "Analyze a song first",
    "optimize.emptyBody": "To optimize parameters, start by analyzing a song first. The optimizer is designed to work from an existing feature set.",
    "optimize.fromUpload": "Analyze with upload",
    "optimize.fromSearch": "Analyze by Spotify ID",
    "optimize.wizardSetup": "Wizard setup",
    "optimize.wizardBody": "Lock the parameters that define the song identity, then free the rest so the backend can search for a better popularity outcome.",
    "optimize.estimatedScore": "Estimated optimized score",
    "optimize.controls": "Optimization controls",
    "optimize.lockPrompt": "Lock what must stay fixed",
    "optimize.run": "Run optimization",
    "optimize.running": "Optimizing...",
    "optimize.finding": "Finding the best configuration",
    "optimize.generateBeat": "Generate AI beat base",
    "optimize.beatGenerating": "Generating AI beat…",
    "optimize.generatedBeatTitle": "Generated beat base",
    "optimize.generatedBeatBody": "Play the generated beat audio. This focuses on the instrumental foundation only.",
    "optimize.viewParams": "View optimized parameters",
    "upload.title": "Upload your audio master",
    "upload.body": "This route supports the producer journey: upload a track, select its genre, process it, and continue into results and optimization.",
    "upload.dragTitle": "Drag and drop a track here",
    "upload.dragBody": "Or click to browse local files.",
    "upload.supported": "Supported: mp3, wav, ogg",
    "upload.genre": "Primary genre",
    "upload.genrePlaceholder": "Select a genre",
    "upload.progress": "Processing progress",
    "upload.status.running": "Running",
    "upload.status.completed": "Completed",
    "upload.status.failed": "Failed",
    "upload.status.waiting": "Waiting",
    "upload.progressBodyLoading": "The bar loops while the backend is processing and completes only when the request returns.",
    "upload.progressBodyIdle": "The result is kept available for the next step once processing succeeds.",
    "upload.selectedFile": "Selected file",
    "upload.analyze": "Analyze my track",
    "upload.analyzing": "Analyzing track...",
    "upload.clear": "Clear",
    "upload.genreRequiredHint": "Genre selection is required before sending the analysis request.",
    "upload.tryAlternate": "Try Spotify ID route",
    "upload.journeyFit": "Journey Fit",
    "upload.journeyBody1": "Best for unreleased songs and producer review sessions before budget or release decisions.",
    "upload.journeyBody2": "After a successful score, the next documented step is the results screen and then the optimization wizard.",
    "upload.flowChecklist": "Flow checklist",
    "upload.step1": "1. Upload a valid audio file.",
    "upload.step2": "2. Select the genre required by the model.",
    "upload.step3": "3. Wait for the backend to complete processing.",
    "upload.step4": "4. Continue into the shared results view.",
    "search.title": "Analyze by Spotify identity",
    "search.body": "Paste a Spotify track ID, URI, or full track URL. This route is built for the artist and A&R lookup journeys.",
    "search.source": "Spotify track source",
    "search.sourcePlaceholder": "Example: 2EKxmYmUdAVXlaHCnnW13o or paste the track URL",
    "search.sourceBody": "Use this route for released songs, references, or catalog analysis.",
    "search.genre": "Genre hint",
    "search.resolve": "Resolve",
    "search.retrieve": "Retrieve",
    "search.predict": "Predict",
    "search.resolveBody": "Spotify reference",
    "search.retrieveBody": "Track features",
    "search.predictBody": "Popularity score",
    "search.analyze": "Analyze Spotify track",
    "search.analyzing": "Analyzing track...",
    "search.clear": "Clear",
    "search.tryAlternate": "Try upload route",
    "search.tips": "Lookup Tips",
    "search.tip1": "Paste the 22-character track ID, a Spotify URI, or a full track link.",
    "search.tip2": "If the ID does not resolve, the backend may not have a matching reference.",
    "search.tip3": "The upload route is the safest option for unreleased or private material.",
    "errors.genreRequiredTitle": "Genre required",
    "errors.genreRequiredBody": "Select a genre before starting the analysis so the model receives a complete feature set.",
    "errors.uploadFileTitle": "Audio file required",
    "errors.uploadFileBody": "Select an audio file before starting the upload analysis flow.",
    "errors.invalidFileTitle": "Invalid file type",
    "errors.invalidFileBody": "Choose a valid audio file in mp3, wav, or ogg format before starting the analysis.",
    "errors.extractTitle": "Audio feature extraction failed",
    "errors.extractBody": "We could not extract the required audio features from this file. Try another audio file or switch to the Spotify ID route.",
    "errors.predictTitle": "Prediction could not be completed",
    "errors.predictBody": "The feature set was created, but the prediction model could not score this track. Try again with a different song or source route.",
    "errors.networkTitle": "Analysis service unavailable",
    "errors.networkBody": "The frontend could not complete the request to the backend service. Confirm that the API is running and try again.",
    "errors.spotifySourceTitle": "Spotify source required",
    "errors.spotifySourceBody": "Enter a Spotify track ID or track URL before starting the lookup route.",
    "errors.spotifyResolveTitle": "Spotify track not found",
    "errors.spotifyResolveBody": "We could not find that Spotify track. Verify the ID or switch to the upload route.",
    "errors.spotifyFeaturesTitle": "Track features unavailable",
    "errors.spotifyFeaturesBody": "The track was found, but the required audio features could not be retrieved.",
    "errors.optimizeTitle": "Optimization failed",
    "errors.optimizeBody": "The backend could not complete the optimization request. Try again or reduce the number of free parameters.",
    "errors.beatGenerationTitle": "Beat generation failed",
    "errors.beatGenerationBody": "The backend could not generate the AI beat base. Try again later.",
    "common.genreSelect": "Select a genre",
    "common.locked": "Locked",
    "common.free": "Free",
    "common.completed": "Completed",
  },
  es: {},
  fr: {},
};

translations.es = {
  ...translations.en,
  "nav.home": "Inicio",
  "nav.analyze": "Analizar",
  "nav.optimize": "Optimizar",
  "nav.how": "Cómo funciona",
  "nav.help": "Ayuda",
  "nav.results": "Resultados",
  "app.startAnalyzing": "Comenzar análisis",
  "app.resumeResult": "Retomar último resultado",
  "app.locale": "Idioma",
  "home.heroTitle": "Predice el potencial de streaming de una canción antes de lanzarla.",
  "home.heroBody": "Un prototipo académico premium para productores, artistas y equipos de A&R. Analiza audio, estima popularidad desde características musicales y explora qué cambiar después.",
  "analyze.title": "Analizar una canción",
  "analyze.body": "Elige la ruta que mejor se ajuste a tu material. Ambas terminan en una vista compartida de resultados y pueden continuar hacia optimización.",
  "how.title": "Cómo funciona",
  "help.title": "Ayuda y limitaciones",
  "results.title": "Resultados del análisis",
  "results.analyzeAnother": "Analizar otra canción",
  "results.whatDoesMean": "¿Qué significa este puntaje?",
  "results.optimize": "Optimizar esta canción",
  "optimize.title": "Optimizar parámetros",
  "optimize.emptyTitle": "Analiza una canción primero",
  "optimize.emptyBody": "Para optimizar parámetros, primero analiza una canción. El optimizador está diseñado para trabajar desde un conjunto de características existente.",
  "optimize.fromUpload": "Analizar con carga",
  "optimize.fromSearch": "Analizar por Spotify ID",
  "upload.title": "Carga tu master de audio",
  "upload.analyze": "Analizar mi canción",
  "upload.analyzing": "Analizando canción...",
  "upload.clear": "Limpiar",
  "upload.tryAlternate": "Probar ruta por Spotify ID",
  "search.title": "Analizar por identidad de Spotify",
  "search.analyze": "Analizar canción de Spotify",
  "search.analyzing": "Analizando canción...",
  "search.clear": "Limpiar",
  "search.tryAlternate": "Probar ruta de carga",
  "errors.genreRequiredTitle": "Género requerido",
  "errors.genreRequiredBody": "Selecciona un género antes de iniciar el análisis para que el modelo reciba un conjunto completo de características.",
  "errors.uploadFileTitle": "Archivo de audio requerido",
  "errors.uploadFileBody": "Selecciona un archivo de audio antes de iniciar la ruta de carga.",
  "errors.invalidFileTitle": "Tipo de archivo inválido",
  "errors.invalidFileBody": "Elige un archivo de audio válido en formato mp3, wav u ogg antes de iniciar el análisis.",
  "errors.extractTitle": "Falló la extracción de características",
  "errors.extractBody": "No pudimos extraer las características de audio necesarias de este archivo. Intenta con otro audio o cambia a la ruta por Spotify ID.",
  "errors.predictTitle": "No se pudo completar la predicción",
  "errors.predictBody": "El conjunto de características fue creado, pero el modelo no pudo puntuar esta canción. Intenta de nuevo con otra canción o con otra ruta.",
  "errors.networkTitle": "Servicio de análisis no disponible",
  "errors.networkBody": "El frontend no pudo completar la solicitud al backend. Verifica que la API esté ejecutándose e inténtalo de nuevo.",
  "errors.spotifySourceTitle": "Fuente de Spotify requerida",
  "errors.spotifySourceBody": "Ingresa un Spotify track ID o una URL de pista antes de iniciar la búsqueda.",
  "errors.spotifyResolveTitle": "No se encontró la canción de Spotify",
  "errors.spotifyResolveBody": "No pudimos encontrar esa canción en Spotify. Verifica el ID o cambia a la ruta de carga.",
  "errors.spotifyFeaturesTitle": "Características no disponibles",
  "errors.spotifyFeaturesBody": "La canción fue encontrada, pero no se pudieron recuperar las características de audio necesarias.",
  "errors.optimizeTitle": "Falló la optimización",
  "errors.optimizeBody": "El backend no pudo completar la solicitud de optimización. Inténtalo de nuevo o reduce la cantidad de parámetros libres.",
  "common.genreSelect": "Selecciona un género",
};

translations.fr = {
  ...translations.en,
  "nav.home": "Accueil",
  "nav.analyze": "Analyser",
  "nav.optimize": "Optimiser",
  "nav.how": "Fonctionnement",
  "nav.help": "Aide",
  "nav.results": "Résultats",
  "app.startAnalyzing": "Commencer l'analyse",
  "app.resumeResult": "Reprendre le dernier résultat",
  "app.locale": "Langue",
  "home.heroTitle": "Prédisez le potentiel de streaming d'une chanson avant sa sortie.",
  "home.heroBody": "Un prototype académique premium pour les producteurs, artistes et équipes A&R. Analysez l'audio, estimez la popularité à partir des caractéristiques musicales et explorez les améliorations possibles.",
  "analyze.title": "Analyser une chanson",
  "analyze.body": "Choisissez la route adaptée à votre source. Les deux parcours mènent à un écran de résultats partagé puis vers l'optimisation.",
  "how.title": "Fonctionnement",
  "help.title": "Aide et limites",
  "results.title": "Résultats de l'analyse",
  "results.analyzeAnother": "Analyser une autre chanson",
  "results.whatDoesMean": "Que signifie ce score ?",
  "results.optimize": "Optimiser cette chanson",
  "optimize.title": "Optimiser les paramètres",
  "optimize.emptyTitle": "Analysez d'abord une chanson",
  "optimize.emptyBody": "Pour optimiser les paramètres, commencez par analyser une chanson. L'optimiseur fonctionne à partir d'un ensemble de caractéristiques existant.",
  "optimize.fromUpload": "Analyser via upload",
  "optimize.fromSearch": "Analyser par Spotify ID",
  "upload.title": "Importez votre master audio",
  "upload.analyze": "Analyser mon morceau",
  "upload.analyzing": "Analyse en cours...",
  "upload.clear": "Effacer",
  "upload.tryAlternate": "Essayer la route Spotify ID",
  "search.title": "Analyser via l'identité Spotify",
  "search.analyze": "Analyser le titre Spotify",
  "search.analyzing": "Analyse en cours...",
  "search.clear": "Effacer",
  "search.tryAlternate": "Essayer la route upload",
  "errors.genreRequiredTitle": "Genre requis",
  "errors.genreRequiredBody": "Sélectionnez un genre avant de lancer l'analyse afin que le modèle reçoive un ensemble de caractéristiques complet.",
  "errors.uploadFileTitle": "Fichier audio requis",
  "errors.uploadFileBody": "Sélectionnez un fichier audio avant de lancer le parcours d'import.",
  "errors.invalidFileTitle": "Type de fichier invalide",
  "errors.invalidFileBody": "Choisissez un fichier audio valide au format mp3, wav ou ogg avant de lancer l'analyse.",
  "errors.extractTitle": "Échec de l'extraction des caractéristiques",
  "errors.extractBody": "Nous n'avons pas pu extraire les caractéristiques audio requises de ce fichier. Essayez un autre fichier audio ou passez à la route Spotify ID.",
  "errors.predictTitle": "La prédiction n'a pas pu être finalisée",
  "errors.predictBody": "L'ensemble de caractéristiques a été créé, mais le modèle n'a pas pu noter ce morceau. Réessayez avec une autre chanson ou une autre route.",
  "errors.networkTitle": "Service d'analyse indisponible",
  "errors.networkBody": "Le frontend n'a pas pu terminer la requête vers le backend. Vérifiez que l'API fonctionne puis réessayez.",
  "errors.spotifySourceTitle": "Source Spotify requise",
  "errors.spotifySourceBody": "Saisissez un identifiant de piste Spotify ou une URL avant de lancer la recherche.",
  "errors.spotifyResolveTitle": "Titre Spotify introuvable",
  "errors.spotifyResolveBody": "Nous n'avons pas trouvé ce titre Spotify. Vérifiez l'identifiant ou passez à la route d'import.",
  "errors.spotifyFeaturesTitle": "Caractéristiques indisponibles",
  "errors.spotifyFeaturesBody": "Le titre a été trouvé, mais les caractéristiques audio requises n'ont pas pu être récupérées.",
  "errors.optimizeTitle": "Échec de l'optimisation",
  "errors.optimizeBody": "Le backend n'a pas pu terminer la demande d'optimisation. Réessayez ou réduisez le nombre de paramètres libres.",
  "common.genreSelect": "Sélectionnez un genre",
};

interface I18nContextValue {
  locale: Locale;
  setLocale: (locale: Locale) => void;
  t: (key: string, vars?: Record<string, string | number>) => string;
}

const I18nContext = createContext<I18nContextValue | null>(null);

function detectLocale(): Locale {
  if (typeof window === "undefined") {
    return "en";
  }

  const saved = window.localStorage.getItem(STORAGE_KEY) as Locale | null;

  if (saved === "en" || saved === "es" || saved === "fr") {
    return saved;
  }

  const languages = window.navigator.languages ?? [window.navigator.language];
  const match = languages
    .map((language) => language.toLowerCase())
    .find((language) => language.startsWith("es") || language.startsWith("fr") || language.startsWith("en"));

  if (match?.startsWith("es")) {
    return "es";
  }

  if (match?.startsWith("fr")) {
    return "fr";
  }

  return "en";
}

export function I18nProvider({ children }: { children: React.ReactNode }): JSX.Element {
  const [locale, setLocale] = useState<Locale>(detectLocale);

  useEffect(() => {
    if (typeof window !== "undefined") {
      window.localStorage.setItem(STORAGE_KEY, locale);
    }
  }, [locale]);

  const value = useMemo<I18nContextValue>(
    () => ({
      locale,
      setLocale,
      t: (key, vars) => {
        const template = translations[locale][key] ?? translations.en[key] ?? key;

        if (!vars) {
          return template;
        }

        return Object.entries(vars).reduce(
          (current, [name, rawValue]) => current.replace(new RegExp(`{${name}}`, "g"), String(rawValue)),
          template
        );
      },
    }),
    [locale]
  );

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>;
}

export function useI18n(): I18nContextValue {
  const context = useContext(I18nContext);

  if (!context) {
    throw new Error("useI18n must be used within an I18nProvider");
  }

  return context;
}
