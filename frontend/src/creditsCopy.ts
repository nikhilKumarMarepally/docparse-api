export function outOfCreditsMessage(creditsPerPage: number, initialCredits: number): string {
  const freePages = Math.max(1, Math.floor(initialCredits / creditsPerPage));
  return `You ran out of credits. Each page costs ${creditsPerPage} credits; new accounts start with ${initialCredits} credits (${freePages} free pages).`;
}

export function signupCreditsMessage(creditsPerPage: number, initialCredits: number): string {
  const freePages = Math.max(1, Math.floor(initialCredits / creditsPerPage));
  return `Sign up with email and password. You get ${initialCredits} free credits (${freePages} pages). Each page uses ${creditsPerPage} credits.`;
}

export function estimateExtractionCost(pageCount: number, creditsPerPage: number): number {
  return Math.max(1, pageCount) * creditsPerPage;
}
